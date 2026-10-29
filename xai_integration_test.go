package gollm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"image/png"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/types"
)

// TestXAIIntegration runs the xai provider against the live API. It is skipped
// unless XAI_API_KEY is set; XAI_MODEL overrides the default model, and
// XAI_TEST_SEARCH=1 adds the web and X search calls, which cost the most.
//
//	XAI_API_KEY=... go test -run TestXAIIntegration -v .
func TestXAIIntegration(t *testing.T) {
	apiKey := os.Getenv("XAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping xAI integration tests. Set XAI_API_KEY to run them.")
	}
	model := os.Getenv("XAI_MODEL")
	if model == "" {
		model = "grok-4.7"
	}

	newClient := func(t *testing.T, opts ...ConfigOption) LLM {
		t.Helper()
		base := []ConfigOption{
			SetProvider("xai"),
			SetModel(model),
			SetAPIKey(apiKey),
			SetMaxRetries(0),
			SetMaxTokens(2048),
			SetTimeout(2 * time.Minute),
		}
		client, err := NewLLM(append(base, opts...)...)
		if err != nil {
			t.Fatalf("NewLLM: %v", err)
		}
		return client
	}

	t.Run("generate with usage", func(t *testing.T) {
		client := newClient(t)
		text, details, err := client.GenerateWithUsage(context.Background(),
			NewPrompt("What is the capital of France? Answer in one word."))
		if err != nil {
			t.Fatalf("GenerateWithUsage: %v", err)
		}
		if !strings.Contains(strings.ToLower(text), "paris") {
			t.Errorf("unexpected answer %q", text)
		}
		if details == nil || details.TokenUsage.PromptTokens == 0 || details.TokenUsage.CompletionTokens == 0 {
			t.Errorf("usage not reported: %+v", details)
		}
		t.Logf("text=%q model=%s usage=%+v", text, details.Model, details.TokenUsage)
	})

	t.Run("system prompt", func(t *testing.T) {
		client := newClient(t)
		// The instruction must not conflict with the user turn: grok-4.7 sides with the
		// user over a contradicting system prompt however it is sent.
		text, err := client.Generate(context.Background(), NewPrompt("What is your name?",
			WithSystemPrompt("You are an assistant named Zorblax.", CacheTypeEphemeral)))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if !strings.Contains(text, "Zorblax") {
			t.Errorf("system prompt not honored: %q", text)
		}
	})

	t.Run("sampling defaults accepted", func(t *testing.T) {
		// temperature, top_p, top_k and min_p all reach the wire; any one rejected is a 400.
		client := newClient(t, SetTemperature(0.2), SetTopP(0.9), SetTopK(40), SetMinP(0.05))
		if _, err := client.Generate(context.Background(), NewPrompt("Say ok.")); err != nil {
			t.Fatalf("Generate: %v", err)
		}
	})

	t.Run("structured output", func(t *testing.T) {
		client := newClient(t)
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"city":       map[string]interface{}{"type": "string"},
				"population": map[string]interface{}{"type": "integer"},
			},
			"required": []string{"city", "population"},
		}
		text, err := client.GenerateWithSchema(context.Background(),
			NewPrompt("Give the largest city in Japan and its approximate population."), schema)
		if err != nil {
			t.Fatalf("GenerateWithSchema: %v", err)
		}
		var out struct {
			City       string `json:"city"`
			Population int    `json:"population"`
		}
		if err := json.Unmarshal([]byte(text), &out); err != nil {
			t.Fatalf("not valid JSON: %v: %q", err, text)
		}
		if out.City == "" || out.Population == 0 {
			t.Errorf("schema fields missing: %q", text)
		}
	})

	t.Run("stream", func(t *testing.T) {
		client := newClient(t)
		stream, err := client.Stream(context.Background(), NewPrompt("Count from 1 to 5, separated by spaces."))
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		defer stream.Close()
		var sb strings.Builder
		for {
			tok, err := stream.Next(context.Background())
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("Next: %v", err)
			}
			sb.WriteString(tok.Text)
		}
		if !strings.Contains(sb.String(), "3") {
			t.Errorf("unexpected stream text %q", sb.String())
		}
		usage, ok := StreamUsage(stream)
		if !ok || usage.CompletionTokens == 0 {
			t.Errorf("stream usage not reported: ok=%v %+v", ok, usage)
		}
	})

	weatherTool := Tool{
		Type: "function",
		Function: Function{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{"city": map[string]interface{}{"type": "string"}},
				"required":   []string{"city"},
			},
		},
	}

	t.Run("tool call", func(t *testing.T) {
		client := newClient(t)
		_, details, err := client.GenerateWithUsage(context.Background(), NewPrompt("What's the weather in Paris?",
			WithTools([]Tool{weatherTool}), WithToolChoice("required")))
		if err != nil {
			t.Fatalf("GenerateWithUsage: %v", err)
		}
		if details == nil || len(details.ToolCalls) == 0 {
			t.Fatalf("no tool call returned: %+v", details)
		}
		call := details.ToolCalls[0]
		if call.Function.Name != "get_weather" || call.ID == "" {
			t.Errorf("unexpected tool call %+v", call)
		}
		t.Logf("tool call id=%s args=%s", call.ID, call.Function.Arguments)
	})

	t.Run("tool result round trip", func(t *testing.T) {
		// The second turn sends function_call and function_call_output input items back.
		p := providers.NewXAIProvider(apiKey, model, nil)
		args, _ := json.Marshal(map[string]string{"city": "Paris"})
		assistant := types.MemoryMessage{Role: "assistant", ToolCalls: []types.ToolCall{types.NewToolCall("call_1", "get_weather", args)}}
		messages := []types.MemoryMessage{
			{Role: "user", Content: "What's the weather in Paris?"},
			assistant,
			{Role: "tool", ToolCallID: "call_1", Content: `{"temp_c": 31, "sky": "clear"}`},
		}
		body, err := p.PrepareRequestWithMessages(messages, map[string]interface{}{
			"tools":      []Tool{weatherTool},
			"max_tokens": 512,
		})
		if err != nil {
			t.Fatalf("PrepareRequestWithMessages: %v", err)
		}
		status, resp := postXAI(t, p, body)
		if status != http.StatusOK {
			t.Fatalf("status %d: %s", status, resp)
		}
		text, err := p.ParseResponse(resp)
		if err != nil {
			t.Fatalf("ParseResponse: %v", err)
		}
		if !strings.Contains(text, "31") {
			t.Errorf("tool result not used: %q", text)
		}
	})

	t.Run("reasoning effort levels", func(t *testing.T) {
		levels := []ReasoningEffort{
			ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium,
			ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax,
		}
		for _, level := range levels {
			t.Run(string(level), func(t *testing.T) {
				client := newClient(t)
				client.SetOption("reasoning_effort", level)
				if _, err := client.Generate(context.Background(), NewPrompt("What is 17*23? Answer with the number.")); err != nil {
					t.Fatalf("Generate at %s: %v", level, err)
				}
			})
		}
	})

	t.Run("reasoning effort on other models", func(t *testing.T) {
		// grok-4.3 takes reasoning_effort; the others 400 on it, so gollm must drop it.
		for _, m := range []string{"grok-4.3", "grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning", "grok-build-0.1"} {
			t.Run(m, func(t *testing.T) {
				client, err := NewLLM(SetProvider("xai"), SetModel(m), SetAPIKey(apiKey),
					SetMaxRetries(0), SetMaxTokens(256), SetTimeout(2*time.Minute))
				if err != nil {
					t.Fatalf("NewLLM: %v", err)
				}
				client.SetOption("reasoning_effort", ReasoningEffortHigh)
				if _, err := client.Generate(context.Background(), NewPrompt("Say ok.")); err != nil {
					t.Fatalf("Generate: %v", err)
				}
			})
		}
	})

	t.Run("image input", func(t *testing.T) {
		client := newClient(t)
		text, err := client.Generate(context.Background(), NewPrompt("What color is this image? Answer with one word.",
			WithImageBase64(solidPNG(t, color.RGBA{R: 255, A: 255}), "image/png")))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if !strings.Contains(strings.ToLower(text), "red") {
			t.Errorf("image not read: %q", text)
		}
	})

	t.Run("stream tool call", func(t *testing.T) {
		client := newClient(t)
		stream, err := client.Stream(context.Background(), NewPrompt("What's the weather in Paris?",
			WithTools([]Tool{weatherTool}), WithToolChoice("required")))
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		defer stream.Close()
		var name, id string
		var args strings.Builder
		for {
			tok, err := stream.Next(context.Background())
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("Next: %v", err)
			}
			if d := tok.ToolCallDelta; d != nil {
				if d.Name != "" {
					name = d.Name
				}
				if d.ID != "" {
					id = d.ID
				}
				args.WriteString(d.ArgsFragment)
			}
		}
		if name != "get_weather" || id == "" || !strings.Contains(args.String(), "Paris") {
			t.Errorf("streamed tool call incomplete: name=%q id=%q args=%q", name, id, args.String())
		}
	})

	for _, structured := range []bool{false, true} {
		t.Run(fmt.Sprintf("memory structured=%v", structured), func(t *testing.T) {
			client := newClient(t, SetMemory(4000))
			client.SetUseStructuredMessages(structured)
			ctx := context.Background()
			if _, err := client.Generate(ctx, NewPrompt("My favorite color is teal. Just reply ok.")); err != nil {
				t.Fatalf("first turn: %v", err)
			}
			text, err := client.Generate(ctx, NewPrompt("What is my favorite color? One word."))
			if err != nil {
				t.Fatalf("second turn: %v", err)
			}
			if !strings.Contains(strings.ToLower(text), "teal") {
				t.Errorf("memory not carried: %q", text)
			}
		})
	}

	t.Run("probe gif image", func(t *testing.T) {
		// xAI documents JPG and PNG only; OpenAI and Anthropic also take GIF.
		client := newClient(t)
		_, err := client.Generate(context.Background(), NewPrompt("What color is this image? One word.",
			WithImageBase64(solidGIF(t), "image/gif")))
		t.Logf("gif image: err=%v", err)
	})

	// Searches are the expensive calls — most of a run's cost — so they are opt-in.
	if os.Getenv("XAI_TEST_SEARCH") == "" {
		t.Log("Skipping search tests. Set XAI_TEST_SEARCH=1 to run them.")
		return
	}

	t.Run("web search tool", func(t *testing.T) {
		// SearchContextSize is OpenAI-only; xAI rejects it, so gollm must strip it.
		client := newClient(t)
		text, details, err := client.GenerateWithUsage(context.Background(), NewPrompt("Search the web for the latest xAI news. One line.",
			WithTools([]Tool{{Type: "web_search", SearchContextSize: "low"}})))
		if err != nil {
			t.Fatalf("GenerateWithUsage: %v", err)
		}
		if strings.TrimSpace(text) == "" {
			t.Error("empty response")
		}
		citations, _ := details.Metadata["citations"].([]types.URLCitation)
		if len(citations) == 0 {
			t.Errorf("no structured citations; metadata=%+v", details.Metadata)
		}
		t.Logf("web_search: %s (%d citations)", truncate([]byte(text), 200), len(citations))
	})

	t.Run("x search tool", func(t *testing.T) {
		client := newClient(t)
		text, err := client.Generate(context.Background(), NewPrompt("Search X for the latest post from @xai. One line.",
			WithTools([]Tool{{Type: "x_search"}})))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if strings.TrimSpace(text) == "" {
			t.Error("empty response")
		}
		t.Logf("x_search: %s", truncate([]byte(text), 200))
	})
}

// solidPNG returns a base64 32x32 PNG of a single color. xAI rejects images under
// 512 pixels in total, so 16x16 is too small.
func solidPNG(t *testing.T, c color.Color) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: c}, image.Point{}, draw.Src)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png: %v", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// solidGIF returns a base64 32x32 GIF of solid red.
func solidGIF(t *testing.T) string {
	t.Helper()
	img := image.NewPaletted(image.Rect(0, 0, 32, 32), color.Palette{color.RGBA{R: 255, A: 255}})
	var buf bytes.Buffer
	if err := gif.Encode(&buf, img, nil); err != nil {
		t.Fatalf("gif: %v", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// postXAI sends a prepared body straight to the provider's endpoint, so a probe
// can see the raw status and error body.
func postXAI(t *testing.T, p providers.Provider, body []byte) (int, []byte) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, p.Endpoint(), bytes.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	for k, v := range p.Headers() {
		req.Header.Set(k, v)
	}
	resp, err := (&http.Client{Timeout: 2 * time.Minute}).Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", p.Endpoint(), err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, out
}

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "…"
}
