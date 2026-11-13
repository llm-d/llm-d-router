// Handler for .github/workflows/pr-close-linked-issues.yaml.
//
// PRs merged through the merge queue are merged by the llm-d-router-release-notes
// App, and GitHub does not close their `Fixes #N` issues (#3703). Human merges
// still close them, so only Bot merges are handled here.

const CLOSING_ISSUES_QUERY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        closingIssuesReferences(first: 50) {
          nodes {
            number
            state
            repository {
              nameWithOwner
            }
          }
        }
      }
    }
  }
`;

async function closeLinkedIssues({ github, context }) {
  const pr = context.payload.pull_request;
  if (!pr?.merged || pr.merged_by?.type !== 'Bot') {
    return [];
  }

  const { owner, repo } = context.repo;
  const resp = await github.graphql(CLOSING_ISSUES_QUERY, {
    owner,
    repo,
    number: pr.number,
  });
  const nodes = resp.repository?.pullRequest?.closingIssuesReferences?.nodes || [];
  // github.token can only write to this repository.
  const toClose = nodes
    .filter((n) => n.state === 'OPEN' && n.repository?.nameWithOwner === `${owner}/${repo}`)
    .map((n) => n.number);

  for (const issue_number of toClose) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number,
      body: `Closing: fixed by #${pr.number}, which was merged through the merge queue.`,
    });
    await github.rest.issues.update({
      owner,
      repo,
      issue_number,
      state: 'closed',
      state_reason: 'completed',
    });
    console.log(`Closed #${issue_number} (linked from #${pr.number}).`);
  }
  return toClose;
}

module.exports = { closeLinkedIssues };
