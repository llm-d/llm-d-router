/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vllmwire

import (
	"crypto/sha256"
	"encoding/hex"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
)

// MediaItem holds the metadata of one GenerateRequest.media item. Byte
// slices alias the scanned payload.
type MediaItem struct {
	// Modality is MediaItem.modality.
	Modality vllmpb.Modality
	// UUID is MediaItem.uuid.
	UUID string
	// HasFeatures reports that the source is PreprocessedMediaFeatures. A
	// later raw source (url, data_uri, raw_bytes) in the same item clears it,
	// as the oneof merge rule requires.
	HasFeatures bool
	// Identifier is features.identifier.
	Identifier []byte
	// HasMMHash reports that features.mm_hash is set.
	HasMMHash bool
	// MMHash is features.mm_hash.
	MMHash []byte
	// Offset and Length are the features placeholder range in the prompt.
	Offset, Length uint64
	// Body is the full MediaItem message body.
	Body []byte
}

// MediaItems returns per-item metadata without copying raw bytes. It
// honors SetMediaBodies.
func (r *Request) MediaItems() ([]MediaItem, error) {
	if r.media.count == 0 {
		return nil, nil
	}
	items := make([]MediaItem, r.media.count)
	for i := range items {
		item, err := parseMediaItem(r.mediaBody(i))
		if err != nil {
			return nil, err
		}
		items[i] = item
	}
	return items, nil
}

func parseMediaItem(body []byte) (MediaItem, error) {
	item := MediaItem{Body: body}
	err := protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		switch num {
		case mediaModality:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			v, _ := protowire.ConsumeVarint(value)
			item.Modality = vllmpb.Modality(int32(uint32(v))) //#nosec G115 -- enum varint truncation matches protobuf
		case mediaUUID:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			item.UUID = string(value)
		case mediaURL, mediaDataURI, mediaRawBytes:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			item.HasFeatures = false
			item.Identifier, item.MMHash, item.HasMMHash = nil, nil, false
			item.Offset, item.Length = 0, 0
		case mediaFeatures:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			item.HasFeatures = true
			return parseFeatures(value, &item)
		}
		return nil
	})
	if err != nil {
		return MediaItem{}, err
	}
	return item, nil
}

func parseFeatures(body []byte, item *MediaItem) error {
	return protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, _ []byte) error {
		switch num {
		case featIdentifier:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			item.Identifier = value
		case featMMHash:
			if wt != protowire.BytesType {
				return ErrMalformed
			}
			item.HasMMHash = true
			item.MMHash = value
		case featOffset:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			item.Offset, _ = protowire.ConsumeVarint(value)
		case featLength:
			if wt != protowire.VarintType {
				return ErrMalformed
			}
			item.Length, _ = protowire.ConsumeVarint(value)
		}
		return nil
	})
}

// ContentDigest returns a SHA-256 hex digest over every wire field of the
// item except uuid. Identical raw media therefore shares one EC key even
// when the client left uuid empty.
func (m MediaItem) ContentDigest() string {
	h := sha256.New()
	_ = protoscan.ForEachField(m.Body, func(num protowire.Number, _ protowire.Type, _, field []byte) error {
		if num != mediaUUID {
			_, _ = h.Write(field)
		}
		return nil
	})
	return hex.EncodeToString(h.Sum(nil))
}

// RewrittenBody returns the item body with uuid replaced by uuidOverride
// (when non-empty) and features.offset replaced by *featuresOffset (when
// set). It returns m.Body itself when there is nothing to rewrite.
func (m MediaItem) RewrittenBody(uuidOverride string, featuresOffset *uint64) []byte {
	if uuidOverride == "" && featuresOffset == nil {
		return m.Body
	}
	out := make([]byte, 0, len(m.Body)+protowire.SizeBytes(len(uuidOverride))+16)
	_ = protoscan.ForEachField(m.Body, func(num protowire.Number, wt protowire.Type, value, field []byte) error {
		switch {
		case uuidOverride != "" && num == mediaUUID:
			// Appended once below so a single uuid field remains.
		case featuresOffset != nil && num == mediaFeatures && wt == protowire.BytesType:
			out = protowire.AppendTag(out, mediaFeatures, protowire.BytesType)
			out = protoscan.AppendLen(out, featuresBodySizeWithOffset(value, *featuresOffset))
			out, _ = protoscan.AppendWithout(out, value, featOffset)
			if *featuresOffset > 0 {
				out = protowire.AppendTag(out, featOffset, protowire.VarintType)
				out = protowire.AppendVarint(out, *featuresOffset)
			}
		default:
			out = append(out, field...)
		}
		return nil
	})
	if uuidOverride != "" {
		out = protowire.AppendTag(out, mediaUUID, protowire.BytesType)
		out = protowire.AppendString(out, uuidOverride)
	}
	return out
}

func featuresBodySizeWithOffset(body []byte, offset uint64) int {
	size, _ := protoscan.SizeWithout(body, featOffset)
	if offset > 0 {
		size += protowire.SizeTag(featOffset) + protowire.SizeVarint(offset)
	}
	return size
}

// mediaMetadataSize returns the size of a MediaItem body after dropping
// url, data_uri, raw_bytes, and features.kwargs.
func mediaMetadataSize(body []byte) int {
	size := 0
	_ = protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, field []byte) error {
		switch {
		case num == mediaURL || num == mediaDataURI || num == mediaRawBytes:
		case num == mediaFeatures && wt == protowire.BytesType:
			inner, _ := protoscan.SizeWithout(value, featKwargs)
			size += protowire.SizeTag(mediaFeatures) + protowire.SizeBytes(inner)
		default:
			size += len(field)
		}
		return nil
	})
	return size
}

// appendMediaMetadata appends the MediaMetadataOnly view of body to dst.
func appendMediaMetadata(dst, body []byte) []byte {
	_ = protoscan.ForEachField(body, func(num protowire.Number, wt protowire.Type, value, field []byte) error {
		switch {
		case num == mediaURL || num == mediaDataURI || num == mediaRawBytes:
		case num == mediaFeatures && wt == protowire.BytesType:
			inner, _ := protoscan.SizeWithout(value, featKwargs)
			dst = protowire.AppendTag(dst, mediaFeatures, protowire.BytesType)
			dst = protoscan.AppendLen(dst, inner)
			dst, _ = protoscan.AppendWithout(dst, value, featKwargs)
		default:
			dst = append(dst, field...)
		}
		return nil
	})
	return dst
}
