# Internal reviewer checklist

Submission status is **NO-GO** while any hard gate is unchecked.

## Hard gates

- [x] Fig. 2 and Table II reproduce from existing raw summaries/case records.
- [x] Case-mean percentile and TTS/proxy circularity are disclosed.
- [ ] Table III comes from a clean, non-saturated, seeded closed-loop rerun with full
  provenance and failure taxonomy.
- [ ] Table IV outcomes follow a pre-registered goal/success/timeout/intervention
  protocol with repeated trials; timing `N` is not presented as trial count.
- [ ] Full-loop timing is measured on named deployment hardware with deadline misses.
- [ ] Abstract, contributions and conclusion contain no forbidden claim.
- [ ] Figures containing people pass privacy/consent review.

## Adversarial reviewer questions

1. Is the quality metric independent of the selection rule? **No; disclose and add
   independent closed-loop metrics.**
2. Why should a single Go Stanford offline set imply deployment robustness? **It
   should not; narrow the claim or add external data.**
3. Are simulation comparisons confounded by stabilizer or duplicated trajectories?
   **Existing data cannot exclude this; rerun.**
4. Are robot outcomes reproducible and statistically supported? **Not yet.**
5. What is algorithmically novel? **At present, little; the defensible novelty is
   systems characterization and evidence discipline. This is a material RA-L risk.**

## Objective verdict

The package is suitable for drafting, not submission. Without new controlled
simulation and repeated robot experiments, likely reviewer objections are not
answerable. Even after those experiments, novelty is moderate rather than strong;
position the work as deployment characterization, or withhold it and prioritize the
future benchmark/method paper.
