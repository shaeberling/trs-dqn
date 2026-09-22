# Second real-experience world update

The second feedback world update completed **16,000 total dynamics updates**,
2,000 beyond its complete 14,000-update parent. All weights, Adam/RNG state,
logs and the immutable data union are preserved here.

The union includes the [continued actor's new collection](../world-model-actor-collection-04/README.md):
**80 training games / 146,625 actions**, with **16 held-out games / 29,367 actions**.
All earlier episodes and held-out labels are unchanged. This is an explicit
own-policy mixture, not an import of evaluation data or demonstrations.

The final uniform held-out graphics MSE is .02038 at one step and .02093 at
24 steps, versus persistence .07188 / .04173. The matching starting audit
at 14,000 is the appropriate comparison; older data populations differ.
On [48 selected loss cases](fit/review-16000/report.json), sixteen-step
continuation Brier remains .98669 versus 1.0 for always surviving.

The subsequent [actor continuation](../imagination-feedback-02/README.md)
regresses substantially. Improved average scene prediction still does not
establish useful control, so the next matched experiment targets multi-step
latent dynamics rather than claiming that feedback alone has solved it.
