// File: utils/shared_types.go
package utils

type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function,omitempty"`
	// Web search parameters (OpenAI & Anthropic)
	Filters           interface{} `json:"filters,omitempty"`             // For web_search: allowed_domains or blocked_domains filter
	UserLocation      interface{} `json:"user_location,omitempty"`       // For web_search: approximate user location
	ExternalWebAccess interface{} `json:"external_web_access,omitempty"` // For web_search (OpenAI): control live internet access
	// Anthropic-specific web search parameters
	MaxUses        interface{} `json:"max_uses,omitempty"`        // For web_search (Anthropic): limit number of searches per request
	AllowedDomains interface{} `json:"allowed_domains,omitempty"` // For web_search: array of allowed domains
	BlockedDomains interface{} `json:"blocked_domains,omitempty"` // For web_search (Anthropic): array of blocked domains
}
