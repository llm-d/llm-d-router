// Package sessionaffinity provides the soft session-affinity scorer. It
// reads the BoundEndpoint attribute published by the session-id-producer
// and gives the bound endpoint a maximum score, leaving every other
// candidate at zero so other scorers still get a vote.
package sessionaffinity
