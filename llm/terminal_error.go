package llm

import (
	"errors"

	"github.com/teilomillet/gollm/types"
)

// isTerminalError reports whether an error should end the retry loop immediately
// rather than be attempted again.
//
// A refusal is the case here: the model declined, and the same prompt will be
// declined again, so the default retry budget turns one refusal into several billed
// calls and then reports only the attempt count.
func isTerminalError(err error) bool {
	return errors.Is(err, types.ErrRefusal)
}
