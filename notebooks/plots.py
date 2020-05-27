import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

with open("correct_probs.pickle", "rb") as f:
    correct_probs = pickle.load(f)

with open("incorrect_probs.pickle", "rb") as f:
    incorrect_probs = pickle.load(f)

xs = np.linspace(0, 1.0, 301)
correct_pdf = st.gaussian_kde(correct_probs).pdf(xs)
incorrect_pdf = st.gaussian_kde(incorrect_probs).pdf(xs)


plt.title("Prediction probability distribution")
plt.xlim(0, 1.0)
plt.plot(xs, correct_pdf, color="green", label="Correct guesses")
plt.plot(xs, incorrect_pdf, color="red", label="Incorrect guesses")
plt.legend()
plt.show()
