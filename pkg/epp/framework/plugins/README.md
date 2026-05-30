# Plugins

This directory contains the available plugins. For detailed information on individual plugins, please refer to the README.md located within each plugin's respective directory. To understand how they integrate into the incoming request processing lifecycle, see the [Endpoint Picker (EPP) design](https://github.com/llm-d/llm-d/tree/main/docs/architecture/core/router/epp) document.

## Session Affinity

Session affinity routes subsequent requests in a session to the same pod as the first request.
Three cooperating plugins implement it:

| Plugin | README |
|--------|--------|
| `session-id-producer` | [requestcontrol/dataproducer/sessionid](requestcontrol/dataproducer/sessionid/README.md) |
| `session-affinity-filter` | [scheduling/filter/sessionaffinity](scheduling/filter/sessionaffinity/README.md) |
| `session-affinity-scorer` | [scheduling/scorer/sessionaffinity](scheduling/scorer/sessionaffinity/README.md) |

## Related Documentation

- [Architecture Overview](../../../../docs/architecture.md)
- [Creating a new Filter guide](../../../../docs/create_new_filter.md)
