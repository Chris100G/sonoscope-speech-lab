# Methods and equations

SpeakReady exposes its practice metrics and keeps the original speech laboratory's NumPy/SciPy reference implementations available in `sonoscope/dsp/` and `sonoscope/recognizers/`.

## Speaking practice

The recording is split into 25 ms frames with 10 ms hops. An RMS-energy threshold relative to the signal peak approximates active speech. Short gaps are joined to reduce fragmented detections. If the user supplies a transcript, pace is estimated as whitespace-delimited tokens divided by estimated active time. The metric is a rough practice aid; it is not transcription or linguistic analysis.

Pitch is estimated via autocorrelation on selected active frames. Only positive periodicity estimates are shown. The middle-half pitch range is expressed in semitones:

\[
R_{\mathrm{pitch}}=12\log_2\left(\frac{Q_{0.75}(F_0)}{Q_{0.25}(F_0)}\right).
\]

## Soft-knee compressor

The RMS detector uses separate one-pole attack and release smoothing. Let \(L\) be the detector level in dBFS, \(T\) the threshold, \(R\) the ratio, and \(K\) the knee width. Above the soft knee, the compressed level is \(T+(L-T)/R\). Within the knee:

\[
L_{\mathrm{out}}=L+\left(\frac{1}{R}-1\right)\frac{(L-T+K/2)^2}{2K}.
\]

The difference between compressed and input level sets the sample gain. This is a feed-forward dynamics compressor, not a brick-wall limiter or denoiser.

## Feature extraction

For a discrete signal \(x[n]\), pre-emphasis is \(y[n]=x[n]-\alpha x[n-1]\). The stream is divided into overlapping frames, multiplied by a finite analysis window, and transformed to a short-time power spectrum. Triangular filters distributed on the mel scale summarize spectral energy; a log transform and orthonormal DCT produce mel-frequency cepstral coefficients (MFCCs). Temporal derivatives are estimated over neighboring frames. Frame length, hop size, filter count, and coefficient count must be recorded for reproducibility.

## LPC and vocal-tract estimates

Linear prediction approximates \(x[n]\) by a weighted sum of past samples. Autocorrelation-based coefficients describe an all-pole vocal-tract model. Roots of its prediction polynomial provide candidate resonances; bandwidth and frequency constraints are needed before interpreting them as formants. Formant estimates are approximate and sensitive to windowing, pitch, and speaker anatomy.

Pitch is estimated from periodicity in voiced frames. Unvoiced speech and silence do not have a meaningful fundamental frequency. Resynthesis is a source-filter illustration and is not intended to reproduce natural voice quality.

## Distances and alignment

Euclidean and cosine distances compare feature vectors, while cepstral/Itakura measures compare spectral-envelope information. DTW finds a minimum-cost monotonic path through two frame sequences:

\[
D(i,j)=d(x_i,y_j)+\min\{D(i-1,j),D(i,j-1),D(i-1,j-1)\}.
\]

Length normalization, allowed path constraints, and score direction must be stated when comparing results. A small DTW cost is not automatically a calibrated confidence.

## Discrete HMM

A discrete HMM consists of initial-state probabilities \(\pi\), transition matrix \(A\), and observation-symbol probabilities \(B\). The forward recurrence sums all paths; the backward recurrence sums future likelihoods; Viterbi selects the maximum-likelihood state path. Baum-Welch applies expectation-maximization to update the parameters from unlabeled sequences. Log probabilities or scaling are required to avoid numerical underflow.

Vector quantization maps continuous feature frames to discrete codebook indices before the discrete HMM stage. LBG-style splitting iteratively refines centroids. Results depend on the codebook size, initialization, enrollment examples, and speaker variation.

## Evaluation

Use held-out recordings and, where possible, held-out speakers. Report command-level accuracy, confusion matrices, rejection behavior, latency, and noise conditions rather than one aggregate number. Synthetic demos establish that the pipeline runs; they are not evidence of real-world recognition accuracy.
