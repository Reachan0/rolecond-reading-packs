# RoleCond gates and scoring parameters

The implementation uses four Boolean admission gates followed by role-specific linear scoring. A rejected candidate receives score `-1`.

These values are engineering choices inherited from the mechanism implementation that preceded the reported audit. The audit holds them fixed; it does not select a best threshold or coefficient using the reported OpenAlex, CSFCube, or presentation outcomes.

## Closely related

Admission:

```text
semantic > 0
OR same_topic
OR same_field
OR same_cluster
```

Score:

```text
1.60 * semantic
+ 1.20 * same_topic
+ 0.90 * same_field
+ 0.70 * same_cluster
+ 0.60 * context_overlap
+ 0.25 * result_overlap
+ 0.25 * reverse_nearest
```

## Methodological neighbors

Admission:

```text
method_overlap >= 0.04
OR (same_field AND semantic >= 0.90)
```

Score:

```text
2.00 * method_overlap
+ 0.80 * same_field
+ 0.45 * same_topic
+ 0.30 * same_discipline
+ 0.25 * semantic
+ 0.15 * same_work_type
```

## Background foundations

Age bonus:

```text
1.0 if older_by >= 5
0.5 if older_by >= 2
0.0 otherwise
```

Admission:

```text
age_bonus > 0
AND (reviewish OR same_field OR same_topic)
```

Score:

```text
0.90 * age_bonus
+ 0.90 * reviewish
+ 0.75 * same_field
+ 0.40 * same_discipline
+ 0.40 * same_topic
+ 0.35 * cited_by_norm
+ 0.20 * context_overlap
```

## Cross-field bridges

Admission:

```text
cross_discipline
AND (
  same_cluster
  OR context_overlap >= 0.05
  OR method_overlap >= 0.05
)
```

Score:

```text
0.90 * cross_discipline
+ 0.80 * same_cluster
+ 0.55 * context_overlap
+ 0.40 * method_overlap
+ 0.25 * semantic
+ 0.25 * cited_by_norm
```

## Assembly

- Canonical role order: closely related, methods, background, cross-field.
- Default per-role budget: 3.
- Within-role ranking: descending role score.
- Deduplication: a paper assigned to an earlier role is unavailable to later roles.
- Sensitivity sweep: budgets 2, 3, and 4; all 24 role orders; fixed, corrected-proportional, and two-stage allocation.
