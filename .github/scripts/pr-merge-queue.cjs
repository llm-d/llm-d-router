// Helpers and handler for .github/workflows/pr-merge-queue.yaml.

const {
  NEEDS_REBASE_LABEL,
  RE_APPROVE_LABEL,
  isHumanReviewer,
  hasApprovalCommand,
} = require('./pr-needs-rebase.cjs');

const BLOCKING_LABELS = [
  'hold',
  'do-not-merge/hold',
  'do-not-merge/work-in-progress',
  NEEDS_REBASE_LABEL,
  RE_APPROVE_LABEL,
];

const PR_STATUS_QUERY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        id
        number
        state
        isDraft
        headRefOid
        mergeable
        mergeStateStatus
        reviewDecision
        mergeQueueEntry {
          id
          position
          state
        }
        labels(first: 100) {
          nodes {
            name
          }
        }
      }
    }
  }
`;

const ENQUEUE_MUTATION = `
  mutation($pullRequestId: ID!) {
    enqueuePullRequest(input: { pullRequestId: $pullRequestId }) {
      mergeQueueEntry {
        id
        position
        state
      }
    }
  }
`;

const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function hasWritePermission(permData) {
  if (!permData) return false;
  if (['admin', 'maintain', 'write'].includes(permData.permission)) {
    return true;
  }
  const perms = permData.user?.permissions;
  return Boolean(perms && (perms.admin || perms.maintain || perms.push));
}

async function resolvePullRequestNumbers({ github, context }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  if (
    context.eventName === 'pull_request_target' ||
    context.eventName === 'pull_request_review'
  ) {
    const number = context.payload.pull_request?.number;
    return number ? [number] : [];
  }

  if (context.eventName === 'workflow_dispatch') {
    const raw = context.payload.inputs?.pr_number;
    const parsed = Number.parseInt(String(raw || ''), 10);
    return Number.isInteger(parsed) && parsed > 0 ? [parsed] : [];
  }

  if (context.eventName === 'issue_comment') {
    const issue = context.payload.issue;
    const comment = context.payload.comment;
    if (!issue || !issue.pull_request || !comment) return [];
    if (!hasApprovalCommand(comment.body)) return [];
    const prAuthor = issue.user?.login;
    if (!isHumanReviewer(comment.user, prAuthor)) {
      console.log(`Ignoring self or bot approval comment by @${comment.user?.login}.`);
      return [];
    }
    const perm = await github.rest.repos.getCollaboratorPermissionLevel({
      owner,
      repo,
      username: comment.user.login,
    });
    if (!hasWritePermission(perm.data)) {
      console.log(
        `User @${comment.user.login} does not have write permission (${perm.data.permission}); skipping.`
      );
      return [];
    }
    return [issue.number];
  }

  if (context.eventName === 'workflow_run') {
    const wr = context.payload.workflow_run;
    if (!wr || wr.conclusion !== 'success') return [];

    const directNumbers = (wr.pull_requests || [])
      .map((pr) => pr.number)
      .filter((n) => Number.isInteger(n) && n > 0);
    if (directNumbers.length > 0) {
      return directNumbers;
    }

    // Fork workflow_run payloads omit pull_requests; resolve via head owner:branch or head_sha.
    const headOwner = wr.head_repository?.owner?.login;
    const headBranch = wr.head_branch;
    const headSha = wr.head_sha;

    if (headOwner && headBranch) {
      const { data: matching } = await github.rest.pulls.list({
        owner,
        repo,
        state: 'open',
        head: `${headOwner}:${headBranch}`,
        per_page: 20,
      });
      const numbers = matching
        .filter((pr) => !headSha || pr.head?.sha === headSha)
        .map((pr) => pr.number);
      if (numbers.length > 0) {
        return numbers;
      }
    }

    if (headSha) {
      const openPrs = await github.paginate(github.rest.pulls.list, {
        owner,
        repo,
        state: 'open',
        per_page: 100,
      });
      return openPrs
        .filter((pr) => pr.head?.sha === headSha)
        .map((pr) => pr.number);
    }
  }

  return [];
}

function getBlockingLabels(pr) {
  const names = (pr.labels?.nodes || []).map((l) => l.name);
  return names.filter((name) => BLOCKING_LABELS.includes(name));
}

async function evaluateAndEnqueue({
  github,
  owner,
  repo,
  pull_number,
  sleep = defaultSleep,
}) {
  let pr = null;
  for (let attempt = 0; attempt < 4; attempt++) {
    const resp = await github.graphql(PR_STATUS_QUERY, {
      owner,
      repo,
      number: pull_number,
    });
    pr = resp.repository?.pullRequest;
    if (!pr) {
      console.log(`PR #${pull_number} not found; skipping.`);
      return { enqueued: false, reason: 'not_found' };
    }

    const blocking = getBlockingLabels(pr);
    const potentiallyReady =
      pr.state === 'OPEN' &&
      !pr.isDraft &&
      !pr.mergeQueueEntry &&
      blocking.length === 0 &&
      pr.reviewDecision === 'APPROVED';

    if (
      pr.mergeable !== 'UNKNOWN' &&
      (!potentiallyReady || pr.mergeStateStatus === 'CLEAN')
    ) {
      break;
    }

    if (attempt < 3) {
      await sleep(3000 * (attempt + 1));
    }
  }

  if (pr.state !== 'OPEN') {
    console.log(`PR #${pull_number} is ${pr.state}; skipping.`);
    return { enqueued: false, reason: 'not_open' };
  }
  if (pr.isDraft) {
    console.log(`PR #${pull_number} is a draft; skipping.`);
    return { enqueued: false, reason: 'draft' };
  }
  if (pr.mergeQueueEntry) {
    console.log(
      `PR #${pull_number} is already in merge queue (position=${pr.mergeQueueEntry.position}, state=${pr.mergeQueueEntry.state}).`
    );
    return { enqueued: false, reason: 'already_queued' };
  }

  const blocking = getBlockingLabels(pr);
  if (blocking.length > 0) {
    console.log(
      `PR #${pull_number} has blocking label(s) [${blocking.join(', ')}]; skipping.`
    );
    return { enqueued: false, reason: 'blocking_label' };
  }

  if (pr.reviewDecision !== 'APPROVED') {
    console.log(
      `PR #${pull_number} reviewDecision is ${pr.reviewDecision || 'NONE'}; skipping.`
    );
    return { enqueued: false, reason: 'not_approved' };
  }

  if (pr.mergeable !== 'MERGEABLE') {
    console.log(`PR #${pull_number} mergeable state is ${pr.mergeable}; skipping.`);
    return { enqueued: false, reason: 'not_mergeable' };
  }

  if (pr.mergeStateStatus !== 'CLEAN') {
    console.log(
      `PR #${pull_number} mergeStateStatus is ${pr.mergeStateStatus}; waiting for all required checks.`
    );
    return { enqueued: false, reason: 'checks_incomplete' };
  }

  try {
    const result = await github.graphql(ENQUEUE_MUTATION, {
      pullRequestId: pr.id,
    });
    const entry = result.enqueuePullRequest?.mergeQueueEntry;
    console.log(
      `Enqueued PR #${pull_number} into merge queue (position=${entry?.position}, state=${entry?.state}).`
    );
    return { enqueued: true, entry };
  } catch (err) {
    const msg = String(err?.message || err);
    if (/already in the merge queue|already queued/i.test(msg)) {
      console.log(`PR #${pull_number} was concurrently added to the merge queue.`);
      return { enqueued: false, reason: 'already_queued' };
    }
    throw err;
  }
}

async function runMergeQueueCheck({ github, context, sleep = defaultSleep }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const numbers = await resolvePullRequestNumbers({ github, context });
  if (numbers.length === 0) {
    console.log(`No eligible pull request identified for event ${context.eventName}.`);
    return;
  }

  for (const pull_number of numbers) {
    await evaluateAndEnqueue({ github, owner, repo, pull_number, sleep });
  }
}

module.exports = {
  BLOCKING_LABELS,
  hasWritePermission,
  resolvePullRequestNumbers,
  getBlockingLabels,
  evaluateAndEnqueue,
  runMergeQueueCheck,
};
