import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


with open("model_verification_stats.pickle", "rb") as f:
    stats = pickle.load(f)


correct_probs = stats["registered"]["accepted_probabilities"]
incorrect_probs = stats["registered"]["rejected_probabilities"]


xs = np.linspace(0, 1.0, 301)
correct_pdf = st.gaussian_kde(correct_probs).pdf(xs)
incorrect_pdf = st.gaussian_kde(incorrect_probs).pdf(xs)


plt.title("Registered users - Identification prediction probability "
          "distribution")
plt.xlim(0, 1.0)
plt.plot(xs, correct_pdf, color="green", label="Correct guesses")
plt.plot(xs, incorrect_pdf, color="red", label="Incorrect guesses")
plt.legend(loc="upper left")
plt.show()


accepted_probs = stats["unknown"]["accepted_probabilities"]
rejected_probs = stats["unknown"]["rejected_probabilities"]

xs = np.linspace(0, 1.0, 301)
correct_pdf = st.gaussian_kde(accepted_probs).pdf(xs)
incorrect_pdf = st.gaussian_kde(rejected_probs).pdf(xs)


plt.title("Unregistered users - Identification prediction probability "
          "distribution")
plt.xlim(0, 1.0)
plt.plot(xs, correct_pdf, color="green", label="Accepted guesses")
plt.plot(xs, incorrect_pdf, color="red", label="Rejected guesses")
plt.legend(loc="upper left")
plt.show()
