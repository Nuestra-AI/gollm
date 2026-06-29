package llm

import (
	"strings"
	"testing"
)

// Fix #8: MaxLineSize is a per-stream option, and the decoder honors it.
func TestWithMaxLineSize(t *testing.T) {
	c := &StreamConfig{}
	WithMaxLineSize(2048)(c)
	if c.MaxLineSize != 2048 {
		t.Errorf("MaxLineSize = %d; want 2048", c.MaxLineSize)
	}
}

func TestSSEDecoderRespectsLimit(t *testing.T) {
	// 100KB single line: exceeds the 64KB cap, fits under 1MB. (bufio floors the
	// effective cap at the 64KB initial buffer, so use a line above that.)
	line := "data: " + strings.Repeat("x", 100_000) + "\n\n"

	small := NewSSEDecoderWithLimit(strings.NewReader(line), 64*1024)
	if small.Next() {
		t.Error("oversized line should not yield an event under a 64KB cap")
	}
	big := NewSSEDecoderWithLimit(strings.NewReader(line), 1<<20)
	if !big.Next() {
		t.Error("line should be read under a 1MB cap")
	}
}
