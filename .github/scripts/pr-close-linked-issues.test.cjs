const assert = require('node:assert/strict');
const { describe, it } = require('node:test');

const { closeLinkedIssues } = require('./pr-close-linked-issues.cjs');

const SAME_REPO = 'llm-d/llm-d-router';

function makeContext({ merged, merged_by }) {
  return {
    repo: { owner: 'llm-d', repo: 'llm-d-router' },
    payload: { pull_request: { number: 42, merged, merged_by } },
  };
}

function ref(number, state, nameWithOwner = SAME_REPO) {
  return { number, state, repository: { nameWithOwner } };
}

function makeGithub(nodes) {
  const comments = [];
  const updates = [];
  const graphqlVars = [];
  const github = {
    graphql: async (query, vars) => {
      graphqlVars.push(vars);
      return { repository: { pullRequest: { closingIssuesReferences: { nodes } } } };
    },
    rest: {
      issues: {
        createComment: async (args) => {
          comments.push(args);
          return { data: {} };
        },
        update: async (args) => {
          updates.push(args);
          return { data: {} };
        },
      },
    },
  };
  return { github, comments, updates, graphqlVars };
}

const bot = { login: 'llm-d-router-release-notes[bot]', type: 'Bot' };
const human = { login: 'octocat', type: 'User' };

const cases = [
  {
    name: 'bot merge closes the single open linked issue',
    pr: { merged: true, merged_by: bot },
    nodes: [ref(7, 'OPEN')],
    want: [7],
  },
  {
    name: 'bot merge closes only open same-repo issues among mixed refs',
    pr: { merged: true, merged_by: bot },
    nodes: [ref(7, 'OPEN'), ref(8, 'CLOSED'), ref(9, 'OPEN', 'llm-d/llm-d'), ref(10, 'OPEN')],
    want: [7, 10],
  },
  {
    name: 'bot merge with no linked issues closes nothing',
    pr: { merged: true, merged_by: bot },
    nodes: [],
    want: [],
  },
  {
    name: 'human merge is left to GitHub and closes nothing',
    pr: { merged: true, merged_by: human },
    nodes: [ref(7, 'OPEN')],
    want: [],
  },
  {
    name: 'PR closed without merge closes nothing',
    pr: { merged: false, merged_by: null },
    nodes: [ref(7, 'OPEN')],
    want: [],
  },
];

describe('closeLinkedIssues', () => {
  for (const tc of cases) {
    it(tc.name, async () => {
      const { github, comments, updates, graphqlVars } = makeGithub(tc.nodes);

      const closed = await closeLinkedIssues({ github, context: makeContext(tc.pr) });

      const sorted = (xs) => [...xs].sort((a, b) => a - b);
      assert.deepEqual(sorted(closed), tc.want);

      const closedViaUpdate = updates
        .filter((u) => u.state === 'closed' && u.owner === 'llm-d' && u.repo === 'llm-d-router')
        .map((u) => u.issue_number);
      assert.deepEqual(sorted(closedViaUpdate), tc.want);
      for (const u of updates) assert.equal(u.state_reason, 'completed');

      const commented = comments.filter((c) => c.body.includes('#42')).map((c) => c.issue_number);
      assert.deepEqual(sorted(commented), tc.want);

      if (tc.want.length === 0) assert.equal(updates.length + comments.length, 0);
      for (const v of graphqlVars) {
        assert.equal(v.owner, 'llm-d');
        assert.equal(v.repo, 'llm-d-router');
        assert.equal(v.number, 42);
      }
    });
  }
});
