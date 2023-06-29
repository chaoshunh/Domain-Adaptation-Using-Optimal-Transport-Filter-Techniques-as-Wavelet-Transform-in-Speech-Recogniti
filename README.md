# Domain Adaptation Using Optimal Transport & Filter Techniques as Wavelet Transform in Speech Recognition**  

Domain adaptation using Optimal Transport as Loss Function using Generative Adversarial Networks on regression and classification problems. <br>
<li>1.- Regression problem: Here we used GAN (Discriminator CNN and Generator a LSTM) & Wavelet Tranform to denoised audio.</li>
<li>2.- Classification problem: Here we used EC-GAN (Discriminador CNN, Generator a LSTM & VGG-Like as classifier).</li><br>
Both cases are using Optimal Transport as Loss Function.

---
**About project**  
---

Speech recognition is a common task in various everyday user systems; however, its effectiveness is limited in noisy environments such as moving vehicles, homes with ambient noise, mobile phones, among others. This work proposes to combine deep learning techniques with domain adaptation and filtering based on Wavelet Transform to eliminate both stationary and non-stationary noise in speech signals in automatic speech recognition (ASR) and speaker identification tasks. It demonstrates how a deep neural network model with domain adaptation, using Optimal Transport, can be trained to mitigate different types of noise. Evaluations were conducted based on Short-Term Objective Intelligibility (STOI) and Perceptual Evaluation of Speech Quality (PESQ). The Wavelet Transform (WT) was applied as a filtering technique to perform a second processing on the speech signal enhanced by the deep neural network, resulting in an average improvement of 20% in STOI and 9% in PESQ compared to the noisy signal. The process was evaluated on a pre-trained ASR system, achieving a general decrease in WER of 14.24%, while an average 99% accuracy in speaker identification. Thus, the proposed approach provides a significant improvement in speech recognition performance by addressing the problem of noisy speech.
