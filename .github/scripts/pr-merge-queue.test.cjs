const assert = require('node:assert/strict');
const { describe, it } = require('node:test');

const {
  resolvePullRequestNumbers,
  evaluateAndEnqueue,
  runMergeQueueCheck,
} = require('./pr-merge-queue.cjs');

function makePr(overrides = {}) {
  return {
    id: 'PR_node_123',
    number: 42,
    state: 'OPEN',
    isDraft: false,
    headRefOid: 'deadbeef',
    mergeable: 'MERGEABLE',
    mergeStateStatus: 'CLEAN',
    reviewDecision: 'APPROVED',
    mergeQueueEntry: null,
    labels: { nodes: [] },
    ...overrides,
  };
}

describe('evaluateAndEnqueue', () => {
  it('enqueues an open non-draft approved PR with CLEAN mergeStateStatus', async () => {
    const calls = [];
    const github = {
      graphql: async (query, vars) => {
        calls.push({ query, vars });
        if (query.includes('enqueuePullRequest')) {
          return {
            enqueuePullRequest: {
              mergeQueueEntry: { id: 'MQE_1', position: 1, state: 'AWAITING_CHECKS' },
            },
          };
        }
        return { repository: { pullRequest: makePr() } };
      },
    };

    const res = await evaluateAndEnqueue({
      github,
      owner: 'llm-d',
      repo: 'llm-d-router',
      pull_number: 42,
      sleep: async () => {},
    });

    assert.equal(res.enqueued, true);
    assert.equal(calls.length, 2);
    assert.deepEqual(calls[1].vars, { pullRequestId: 'PR_node_123' });
  });

  for (const label of [
    'hold',
    'do-not-merge/hold',
    'do-not-merge/work-in-progress',
    'needs-rebase',
    're-approve',
  ]) {
    it(`skips PR carrying blocking label ${label}`, async () => {
      const github = {
        graphql: async () => ({
          repository: {
            pullRequest: makePr({ labels: { nodes: [{ name: label }] } }),
          },
        }),
      };

      const res = await evaluateAndEnqueue({
        github,
        owner: 'llm-d',
        repo: 'llm-d-router',
        pull_number: 42,
        sleep: async () => {},
      });

      assert.equal(res.enqueued, false);
      assert.equal(res.reason, 'blocking_label');
    });
  }

  it('skips PR that is not approved', async () => {
    const github = {
      graphql: async () => ({
        repository: { pullRequest: makePr({ reviewDecision: 'REVIEW_REQUIRED' }) },
      }),
    };

    const res = await evaluateAndEnqueue({
      github,
      owner: 'llm-d',
      repo: 'llm-d-router',
      pull_number: 42,
      sleep: async () => {},
    });

    assert.equal(res.enqueued, false);
    assert.equal(res.reason, 'not_approved');
  });

  it('skips PR already in the merge queue', async () => {
    const github = {
      graphql: async () => ({
        repository: {
          pullRequest: makePr({
            mergeQueueEntry: { id: 'MQE_1', position: 1, state: 'QUEUED' },
          }),
        },
      }),
    };

    const res = await evaluateAndEnqueue({
      github,
      owner: 'llm-d',
      repo: 'llm-d-router',
      pull_number: 42,
      sleep: async () => {},
    });

    assert.equal(res.enqueued, false);
    assert.equal(res.reason, 'already_queued');
  });

  it('retries transient BLOCKED state when approved and enqueues once CLEAN', async () => {
    let fetchCount = 0;
    const github = {
      graphql: async (query) => {
        if (query.includes('enqueuePullRequest')) {
          return {
            enqueuePullRequest: {
              mergeQueueEntry: { id: 'MQE_2', position: 2, state: 'QUEUED' },
            },
          };
        }
        fetchCount++;
        return {
          repository: {
            pullRequest: makePr({
              mergeStateStatus: fetchCount < 2 ? 'BLOCKED' : 'CLEAN',
            }),
          },
        };
      },
    };

    const res = await evaluateAndEnqueue({
      github,
      owner: 'llm-d',
      repo: 'llm-d-router',
      pull_number: 42,
      sleep: async () => {},
    });

    assert.equal(fetchCount, 2);
    assert.equal(res.enqueued, true);
  });
});

describe('resolvePullRequestNumbers', () => {
  it('resolves fork PR from workflow_run via head owner:branch and head_sha', async () => {
    const github = {
      rest: {
        pulls: {
          list: async ({ head }) => {
            assert.equal(head, 'contributor:feat-branch');
            return {
              data: [
                { number: 99, head: { sha: 'other-sha' } },
                { number: 100, head: { sha: 'target-sha' } },
              ],
            };
          },
        },
      },
    };
    const context = {
      eventName: 'workflow_run',
      repo: { owner: 'llm-d', repo: 'llm-d-router' },
      payload: {
        workflow_run: {
          conclusion: 'success',
          pull_requests: [],
          head_repository: { owner: { login: 'contributor' } },
          head_branch: 'feat-branch',
          head_sha: 'target-sha',
        },
      },
    };

    const nums = await resolvePullRequestNumbers({ github, context });
    assert.deepEqual(nums, [100]);
  });

  it('validates collaborator permission and approval command on issue_comment', async () => {
    const github = {
      rest: {
        repos: {
          getCollaboratorPermissionLevel: async () => ({
            data: { permission: 'custom', user: { permissions: { push: true } } },
          }),
        },
      },
    };
    const context = {
      eventName: 'issue_comment',
      repo: { owner: 'llm-d', repo: 'llm-d-router' },
      payload: {
        issue: { number: 77, pull_request: {}, user: { login: 'author' } },
        comment: { body: '/lgtm', user: { login: 'reviewer', type: 'User' } },
      },
    };

    const nums = await resolvePullRequestNumbers({ github, context });
    assert.deepEqual(nums, [77]);
  });

  it('runs end-to-end via runMergeQueueCheck on pull_request_target', async () => {
    let enqueuedId = null;
    const github = {
      graphql: async (query, vars) => {
        if (query.includes('enqueuePullRequest')) {
          enqueuedId = vars.pullRequestId;
          return {
            enqueuePullRequest: {
              mergeQueueEntry: { id: 'MQE_3', position: 1, state: 'QUEUED' },
            },
          };
        }
        return { repository: { pullRequest: makePr({ id: 'PR_target_1', number: 55 }) } };
      },
    };
    const context = {
      eventName: 'pull_request_target',
      repo: { owner: 'llm-d', repo: 'llm-d-router' },
      payload: { pull_request: { number: 55 } },
    };

    await runMergeQueueCheck({ github, context, sleep: async () => {} });
    assert.equal(enqueuedId, 'PR_target_1');
  });
});
