package types

import "errors"

// ErrRefusal marks a response in which the model declined to answer. A refusal is
// a completed, billed call that carries a reason instead of a completion, so it is
// reported as an error — but a deterministic one: retrying spends money to be
// refused again. Providers wrap their refusal text with it; retry loops treat it
// as terminal.
var ErrRefusal = errors.New("model refused the request")
