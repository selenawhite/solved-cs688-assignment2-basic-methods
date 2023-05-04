Download Link: https://assignmentchef.com/product/solved-cs688-assignment2-basic-methods
<br>
Introduction: In this assignment, you will experiment with different aspects of modeling, learning, and inference with chain-structured conditional random fields (CRFs). This assignment focuses on the task of optical character recognition (OCR). We will explore an approach that bridges computer vision and natural language processing by jointly modeling the labels of sequences of noisy character images that form complete words. This is a natural problem for chain-structured CRFs. The node potentials can capture bottom-up information about the character represented by each image, while the edge potentials can capture information about the co-occurrence of characters in adjacent positions within a word.

Data: The underlying data are a set of <em>N </em>sequences corresponding to images of the characters in individual words. Each word <em>i </em>consists of <em>L<sub>i </sub></em>positions. For each position <em>j </em>in word <em>i</em>, we have a noisy binary image of the character in the that position. In this assignment, we will use the raw pixel values of the character images as features in the CRF. The character images are 20 × 16 pixels. We convert them into 1 × 320 vectors. We include a constant bias feature along with the pixels in each image, giving a final feature vector of length <em>F </em>= 321. <em>x<sub>ijf </sub></em>indicates the value of feature <em>f </em>in position <em>j </em>of word <em>i</em>. The provided training and test files <em>train_img&lt;i&gt;.txt </em>and <em>test_img&lt;i&gt;.txt </em>list the character image <strong>x</strong><em><sub>ij </sub></em>on row <em>j </em>of file <em>i </em>as a 321-long, space-separated sequence.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> The data files are in the column-major format. Given the sequence of character images <strong>x</strong><em><sub>i </sub></em>= [<strong>x</strong><em><sub>i</sub></em><sub>1</sub><em>,…,</em><strong>x</strong><em><sub>iL</sub></em><em><sub>i</sub></em>] corresponding to test word <em>i</em>, your goal is to infer the corresponding sequence of character labels <strong>y</strong><em><sub>i </sub></em>= [<em>y<sub>i</sub></em><sub>1</sub><em>,…,y<sub>iL</sub></em><em><sub>i</sub></em>]. There are <em>C </em>= 26 possible labels corresponding to the lower case letters “a” to “z.” The character labels for each training and test word are available in the files <em>train_words.txt </em>and <em>test_words.txt</em>. The figure below shows several example words along with their images.

shoot                                        indoor                                       threee                                         trait

Model: The conditional random field model is a conditional model <em>P<sub>W</sub></em>(<strong>y</strong><em><sub>i</sub></em>|<strong>x</strong><em><sub>i</sub></em>) of the sequence of class labels <strong>y</strong><em><sub>i </sub></em>given the sequence of feature vectors <strong>x</strong><em><sub>i </sub></em>that depends on a collection of parameters <em>W</em>. The CRF graphical model is shown below for a sequence of length 4.

Conditional Random Field

The probabilistic model for the CRF we use in this assignment is given below. The CRF model contains one feature parameter <em>W<sub>cf</sub><sup>F </sup></em>for each of the <em>C </em>labels and <em>F </em>features. The feature parameters encode the compatibility between feature values and labels. The CRF also contains one transition parameter for each pair of labels <em>c </em>and <em>c</em><sup>0</sup>. The transition parameters encode the compatibility between adjacent labels in the sequence. We parameterize the model in log-space, so all of the parameters can take arbitrary (positive or negative) real values. We have one feature potential <em>φ<sup>F</sup><sub>j </sub></em>(<em>y<sub>ij</sub>,</em><strong>x</strong><em><sub>ij</sub></em>) for each position <em>j </em>in sequence <em>i </em>and one transition potential for each pair of adjacent labels <em>φ<sup>T</sup><sub>j </sub></em>(<em>y<sub>ij</sub>,y<sub>ij</sub></em><sub>+1</sub>) in sequence <em>i</em>.

<em>C         F</em>

<em>φ</em><em>Fj </em>(<em>y</em><em>ij</em><em>,</em><strong>x</strong><em>ij</em>) = XX<em>W</em><em>cfF </em>[<em>y</em><em>ij </em>= <em>c</em>]<em>x</em><em>ijf</em>

<em>c</em>=1 <em>f</em>=1

<em>C         C</em>

<em>φ</em><em>Tj </em>(<em>y</em><em>ij</em><em>,y</em><em>ij</em>+1) = XX <em>W</em><em>ccT</em>0[<em>y</em><em>ij </em>= <em>c</em>][<em>y</em><em>ij</em>+1 = <em>c</em>0]

<em>c</em>=1 <em>c</em><sup>0</sup>=1

Given this collection of potentials, the joint energy function on <strong>x</strong><em><sub>i </sub></em>and <strong>y</strong><em><sub>i </sub></em>is defined below.

 <em>L</em><em>i                                                    </em><em>L</em><em>i</em>−1                                              

<em>E</em><em>W</em>(<strong>y</strong><em>i</em><em>,</em><strong>x</strong><em>i</em>) = −X<em>φ</em><em>Fj </em>(<em>y</em><em>ij</em><em>,</em><strong>x</strong><em>ij</em>) + X <em>φ</em><em>Tj </em>(<em>y</em><em>ij</em><em>,y</em><em>ij</em>+1)

<em>j</em>=1                                                  <em>j</em>=1

<sup> </sup><em>L<sub>i          </sub>C         F                                                                     L<sub>i</sub></em>−1 <em>C             C                                                                                </em><sup></sup>

= −XXX<em>W</em><em>cfF </em>[<em>y</em><em>ij </em>= <em>c</em>]<em>x</em><em>ijf </em>+ XXX <em>W</em><em>ccT</em>0[<em>y</em><em>ij </em>= <em>c</em>][<em>y</em><em>ij</em>+1 = <em>c</em>0]

<em>j</em>=1 <em>c</em>=1 <em>f</em>=1                                                                      <em>j</em>=1 <em>c</em>=1 <em>c</em><sup>0</sup>=1

The joint probability of <strong>y</strong><em><sub>i </sub></em>and <strong>x</strong><em><sub>i </sub></em>is given by the Gibbs distribution for the model.

<strong>y x</strong>

However, as the name implies, a conditional random field is not trained to maximize the joint likelihood of <strong>x </strong>and <strong>y</strong>. Instead, the model is trained to maximize the conditional likelihood of <strong>y </strong>given <strong>x </strong>similar to a discriminative classifier like logistic regression. The conditional probability of <strong>y </strong>given <strong>x </strong>is shown below. Note that the partition function <em>Z<sub>W</sub></em>(<strong>x</strong><em><sub>i</sub></em>) that results from conditioning on a sequences of feature vectors <strong>x</strong><em><sub>i </sub></em>will generally be different for each sequence <strong>x</strong><em><sub>i</sub></em>.

<em>,</em><strong>x</strong><em><sub>i</sub></em>))

<strong>y</strong>

<ol>

 <li>(<em>10 points</em>) Basics: To begin, implement the following basic methods. While we will only experiment with one data set, your code should be written to work with any label set and any number of features.

  <ul>

   <li>(<em>2 pts</em>) Implement the function get_params, which returns the current model parmeters.</li>

   <li>(<em>2 pts</em>) Implement the function set_params, which sets the current model parmeters.</li>

   <li>(<em>6 pts</em>) Implement the function energy, which computes the joint energy of a label and feature sequence <strong>y </strong>and <strong>x</strong>.</li>

  </ul></li>

 <li>(<em>30 points</em>) Inference: Efficient inference is the key to both learning and prediction in CRFs. In this question you will describe and implement inference methods for chain-structured CRF.

  <ul>

   <li>(<em>10 pts</em>) Explain how factor reduction and the log-space sum-product message passing algorithms can be combined to enable efficient inference for the single-node distributions <em>P<sub>W</sub></em>(<em>y<sub>j</sub></em>|<strong>x</strong>) and the pairwise distribution <em>P<sub>W</sub></em>(<em>y<sub>j</sub>,y<sub>j</sub></em><sub>+1</sub>|<strong>x</strong>). These distributions are technically conditional marginal distributions, but since we are always conditioning on <strong>x </strong>in a CRF, we will simply refer to them as marginal, and pairwise marginal distributions. Your solution to this question must have linear complexity in the length of the input, and should not numerically underflow or overflow even for long sequences and many labels. (report)</li>

   <li>(<em>10 pts</em>) Implement the function log_Z, which computes the log partition function for the distributin <em>P<sub>W</sub></em>(<strong>y</strong>|<strong>x</strong>). (code)</li>

   <li>(<em>10 pts</em>) Implement the function predict_logprob, which computes the individual <em>P<sub>W</sub></em>(<em>y<sub>j</sub></em>|<strong>x</strong>) and pairwise marginals <em>P<sub>W</sub></em>(<em>y<sub>j</sub>,y<sub>j</sub></em><sub>+1</sub>|<strong>x</strong>) for each position in the sequence. (code)</li>

  </ul></li>

 <li>(<em>30 points</em>) Learning: In this problem, you will derive the maximum likelihood learning algorithm for chain-structured conditional random field models. Again, this algorithm maximizes the average conditional log likelihood function , not the average joint log likelihood, but the learning approach is still typically referred to as maximum likelihood.

  <ul>

   <li>(<em>2 pts</em>) Write down the average conditional log likelihood function for the CRF given a data set consisting of <em>N </em>input sequences <strong>x</strong><em><sub>i </sub></em>and label sequences <strong>y</strong><em><sub>i </sub></em>in terms of the parameters and the data. (report)</li>

   <li>(<em>5 pts</em>) Derive the derivative of the average conditional log likelihood function with respect to the feature parameter <em>W<sub>cf</sub><sup>F </sup></em>. Show your work. (report)</li>

   <li>(<em>5 pts</em>) Derive the derivative of the average conditional log likelihood function with respect to the transition parameter. Show your work. (report)</li>

   <li>(<em>3 pts</em>) Explain how the average conditional log likelihood function and its gradients can be efficiently computed using the inference method you developed in the previous question. (report)</li>

   <li>(<em>5 pts</em>) Implement the function log_likelihood to efficiently compute the average conditional log likelihood given a set of labeled input sequences. (code)</li>

   <li>(<em>5 pts</em>) Implement the function gradient_log_likelihood to efficiently compute the gradient of the average conditional log likelihood given a set of labeled input sequences. (code)</li>

   <li>(<em>5 pts</em>) Use your implementation of log_likelihood and gradient_log_likelihood along with a numerical optimizer to implement maximum (conditional) likelihood learning in the fit function. The reference solutions were computed using the fmin_bfgs method from scipy.optimize using default optimizer settings. It is recommended that you use this optimizer and the default settings as well to minimize any discrepancies.</li>

  </ul></li>

</ol>

(code)

<ol start="4">

 <li>(<em>10 points</em>) Prediction: To use the learned model to make predictions, implement the function predict. Given an unlabeled input sequence, this function should compute the node marginal <em>P<sub>W</sub></em>(<em>y<sub>j</sub></em>|<strong>x</strong>) for every position <em>j </em>in the label sequence conditioned on a feature sequence <strong>x</strong>, and then predict the marginally most likely label. This is called <em>max marginal prediction</em>. (code).</li>

 <li>(<em>20 points</em>) Experiments: In this problem, you will use your implementation to conduct basic learning experiments. Add your experiment code to experiment.py

  <ul>

   <li>(<em>10 pts</em>) Use your CRF implementation and the first 100, 200, 300, 400, 500, 600, 700, and 800 training cases to learn eight separate models. For each model, compute the average test set conditional log likelihood and the average test set prediction error. As your answer to this question, provide two separate line plots showing average test set conditional log likelihood and average test error vs the number of training cases. (report)</li>

   <li>(<em>10 pts</em>) Using your CRF model trained on all 800 data cases, conduct an experiment to see if the compute time needed to perform max marginal inference scales linearly with the length of the feature sequence as expected. You should experiment with sequences up to length 20. You will need to create your own longer sequences. Explain how you did so. You should also use multiple repetitions to stabilize the time estimates. As your answer to this question, provide a line plot showing the average time needed to perform marginal inference vs the sequence length. (report)</li>

  </ul></li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> Images are also provided for each training and test word as standard PNG-format files <em>train_img&lt;i&gt;.png </em>and <em>test_img&lt;i&gt;.png</em>. These are for your reference and not for use in training or testing algorithms.