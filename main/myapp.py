# streamlit run myapp.py

# 11.17.21 Wed
# jumpstarted
# worked on imgcmp.py
# incorporated imgcmp.py into myapp.py
# learned st.pyplot !!!
# Next step: align the output images in a better way. 

# 

import streamlit as st
from PIL import Image

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os


### Excluding Imports ###
st.title("Approximation of Images by Singular Value Decomposition")

uploaded_file = st.file_uploader("Choose an image.", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Convert RGB to grayscale")
        
    A = image
    A = np.asarray(A)
    # print(A.shape)
    # Convert RGB to grayscale.
    X = np.mean(A,-1)
    # print(X.shape)

    # Save the gray image.
    fig, ax = plt.subplots()
    img = plt.imshow(X)
    img.set_cmap('gray')
    ax.axis('off')
    plt.savefig('dog_gray.jpg')
    plt.rcParams['figure.figsize'] = [10, 8]
    st.pyplot(fig)
    plt.close()


    U, S, VT = np.linalg.svd(X,full_matrices=False)
    S = np.diag(S)
    # print(40*'-.')
    # print(U.shape)
    # print(S.shape)
    # print(VT.shape)
    # print(40*'-.')

    j = 0
    for r in (5, 20, 100):
        # Construct approximate image
        Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
        fig = plt.figure(j+1)
        j += 1
        img = plt.imshow(Xapprox)
        img.set_cmap('gray')
        plt.axis('off')
        plt.title('r = ' + str(r))
        st.pyplot(fig)
        plt.savefig('dog_gray_{}.jpg'.format(r))
        plt.close()



    # Analysis of singular values.
    fig = plt.figure(1)
    plt.semilogy(np.diag(S))
    plt.title('Singular Values')
    st.pyplot(fig)
    plt.savefig('singularVal.png')
    plt.close()

    fig = plt.figure(2)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    plt.title('Singular Values: Cumulative Sum')
    st.pyplot(fig)
    plt.savefig('singularValCumul.png')
    plt.close()
