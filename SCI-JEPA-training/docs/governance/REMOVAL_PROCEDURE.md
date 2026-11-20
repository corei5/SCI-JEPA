# Removal procedure (GDPR Art. 17, 21; authors' requests)

1. Requests reach the contact in release/CONTACT.
2. Add the paper id to release/exclude_papers.txt with the reason and date.
3. Releases: `sci-jepa release export ...` drops excluded papers from any KU text file at once.
4. Training data: excluded papers are dropped when the split / index is rebuilt for the next training run; published weights are replaced at the next release (weights contain no personal data).
5. Answer the requester within one month (Art. 12(3)).

## Sign-off

Signed off by: ____________________
Role: ____________________
Date: ____________________
