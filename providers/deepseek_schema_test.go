package providers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/teilomillet/gollm/types"
)

// DeepSeek's response_format accepts only "text" or "json_object", so a schema request goes out as
// json_object with the schema in the system prompt, never as json_schema. The caller's system prompt
// is kept ahead of the instruction, and the caller's options map is not modified.
func TestDeepSeekSchemaRequestsUseJSONObject(t *testing.T) {
	p := NewDeepSeekProvider("k", "deepseek-chat", nil)
	schema := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{"q": map[string]interface{}{"type": "string", "minLength": 3}},
		"required":   []string{"q"},
	}

	requests := []struct {
		name    string
		prepare func(options map[string]interface{}) ([]byte, error)
	}{
		{"single prompt", func(options map[string]interface{}) ([]byte, error) {
			return p.PrepareRequestWithSchema("hi", options, schema)
		}},
		{"messages", func(options map[string]interface{}) ([]byte, error) {
			return p.PrepareRequestWithMessagesAndSchema([]types.MemoryMessage{{Role: "user", Content: "hi"}}, options, schema)
		}},
	}
	for _, r := range requests {
		t.Run(r.name, func(t *testing.T) {
			options := map[string]interface{}{"system_prompt": "Be brief."}
			body, err := r.prepare(options)
			require.NoError(t, err)
			assert.Equal(t, "Be brief.", options["system_prompt"], "caller's options modified")

			var request struct {
				ResponseFormat map[string]interface{} `json:"response_format"`
				Messages       []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(body, &request))

			assert.Equal(t, map[string]interface{}{"type": "json_object"}, request.ResponseFormat)
			assert.NotContains(t, string(body), "json_schema")

			require.NotEmpty(t, request.Messages)
			system := request.Messages[0]
			assert.Equal(t, "system", system.Role)
			assert.True(t, strings.HasPrefix(system.Content, "Be brief.\n\n"), "caller's system prompt first: %q", system.Content)
			assert.Contains(t, system.Content, "JSON")
			assert.Contains(t, system.Content, `"required":["q"]`)
			assert.Equal(t, "user", request.Messages[len(request.Messages)-1].Role)
		})
	}
}
