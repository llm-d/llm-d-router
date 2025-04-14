package kvcache

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/neuralmagic/distributed-kv-cache/pkg/utils"
	"github.com/redis/go-redis/v9"
	"golang.org/x/net/context"
)

// KVBlockIndexerConfig holds the configuration for the KVBlockIndexer.
type KVBlockIndexerConfig struct {
	*RedisKVBlockIndexerConfig
}

// DefaultKVBlockIndexerConfig returns the default configuration for the KVBlockIndexer.
func DefaultKVBlockIndexerConfig() *KVBlockIndexerConfig {
	return &KVBlockIndexerConfig{
		RedisKVBlockIndexerConfig: defaultRedisKVBlockIndexerConfig(),
	}
}

// KVBlockIndexer defines the interactions with the KVCache indexing backend.
type KVBlockIndexer interface {
	// GetPodsForKeys receives a list of keys and a set of pod identifiers,
	// and retrieves the filtered pods associated with those keys.
	// The filtering is done based on the pod identifiers provided.
	// If the podIdentifierSet is empty, all pods are returned.
	//
	// It returns:
	// 1. A slice of strings representing the keys.
	// 2. A map where the keys are those in (1) and the values are pod names.
	// 3. An error if any occurred during the operation.
	GetPodsForKeys(ctx context.Context,
		keys []KVBlockKey, podIdentifierSet sets.Set[string]) ([]string, map[string]string, error)
}

var _ KVBlockIndexer = &RedisKVBlockIndexer{}

// RedisKVBlockIndexerConfig holds the configuration for the RedisKVBlockIndexer.
type RedisKVBlockIndexerConfig struct {
	RedisAddr     string
	RedisPassword string
	RedisDB       int
}

func defaultRedisKVBlockIndexerConfig() *RedisKVBlockIndexerConfig {
	return &RedisKVBlockIndexerConfig{
		RedisAddr:     "localhost:6379",
		RedisPassword: "",
		RedisDB:       0,
	}
}

type RedisKVBlockIndexer struct {
	// RedisClient is the Redis client used for communication.
	RedisClient *redis.Client
}

// NewRedisKVBlockIndexer creates a new RedisKVBlockIndexer instance.
func NewRedisKVBlockIndexer(config *KVBlockIndexerConfig) (KVBlockIndexer, error) {
	if config == nil {
		config = DefaultKVBlockIndexerConfig()
	}

	redisClient := redis.NewClient(&redis.Options{
		Addr:     config.RedisAddr,
		Password: config.RedisPassword,
		DB:       config.RedisDB,
	})

	_, err := redisClient.Ping(context.Background()).Result()
	if err != nil {
		return nil, fmt.Errorf("could not connect to Redis: %w", err)
	}

	return &RedisKVBlockIndexer{
		RedisClient: redisClient,
	}, nil
}

// GetPodsForKeys receives a list of keys and a set of pod identifiers,
// and retrieves the filtered pods associated with those keys.
// The filtering is done based on the pod identifiers provided.
// If the podIdentifierSet is empty, all pods are returned.
//
// It returns:
// 1. A slice of strings representing the keys.
// 2. A map where the keys are those in (1) and the values are pod names.
// 3. An error if any occurred during the operation.
//
//nolint:gocritic // no need named return values here
func (r *RedisKVBlockIndexer) GetPodsForKeys(ctx context.Context,
	keys []KVBlockKey, podIdentifierSet sets.Set[string],
) ([]string, map[string]string, error) {
	pods := make(map[string]string)

	redisKeys := utils.SliceMap(keys, func(key KVBlockKey) string {
		return key.String()
	})
	// use redis.MGet to get all keys at once
	values, err := r.RedisClient.MGet(ctx, redisKeys...).Result()
	if err != nil {
		return nil, nil, err
	}

	filterPods := len(podIdentifierSet) > 0

	for i, value := range values { // values are "podIP:port", we only need podIP
		if value == "" {
			continue
		}

		valueStr, ok := value.(string)
		if !ok {
			continue
		}

		parts := strings.Split(valueStr, ":")
		if len(parts) != 2 {
			continue
		}

		// check if the pod identifier is in the set
		if filterPods && !podIdentifierSet.Has(parts[0]) {
			continue
		}

		pods[redisKeys[i]] = parts[0]
	}

	return redisKeys, pods, nil
}
