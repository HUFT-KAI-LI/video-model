"""Discrete first-failure risk for [left, right), with unresolved labels explicit."""


def interval_hazards(videos, reviews, edges=(0, 15, 30, 45, 60)):
    from summarize import prefix_status
    result = []
    for left, right in zip(edges, edges[1:]):
        eligible = [v for v in videos if v['duration_sec'] >= right]
        at_risk = prior = unknown_entry = failures = survivors = unresolved = 0
        for video in eligible:
            review = reviews[video['video_id']]
            # All enrolled trajectories are at risk immediately before their first frame.
            entry = 'normal' if left == 0 else prefix_status(review, left)
            if entry == 'failure':
                prior += 1
            elif entry == 'unknown':
                unknown_entry += 1
            else:
                at_risk += 1
                end = prefix_status(review, right)
                if end == 'failure':
                    failures += 1
                elif end == 'normal':
                    survivors += 1
                else:
                    unresolved += 1
        resolved = unknown_entry == 0 and unresolved == 0
        result.append(dict(start_sec=left, end_sec=right, n_eligible=len(eligible),
                           n_prior_failure=prior, n_at_risk=at_risk,
                           n_unknown_entry=unknown_entry, n_first_failures=failures,
                           n_survived=survivors, n_unresolved_outcome=unresolved,
                           hazard=failures / at_risk if resolved and at_risk else None,
                           status='resolved' if resolved and at_risk else
                                  ('empty_risk_set' if resolved else 'pending_review')))
    return result
