# vision-transformer-triplet-loss-classification
This repository implements a Vision Transformer for classifying images using triplet loss, focusing on detecting Waldo from the "Where's Waldo" series.



\section*{Introduction}
\textit{Where's Waldo?} is a children's book series and game. Readers search for Waldo, a tall man in a red striped shirt, red beanie, and glasses, hidden among busy scenes. The goal is to find Waldo as quickly as possible. For this project, we obtained a dataset from the Hey-Waldo repository on GitHub.

\section*{Code Explanation}

\subsection*{Imports and Setup}
\begin{itemize}
    \item The necessary libraries for deep learning (PyTorch), image processing, and data manipulation are imported.
    \item The random seeds are set for reproducibility.
    \item The device (CPU or GPU) is selected for training.
\end{itemize}

\subsection*{Data Loading}
\begin{itemize}
    \item A custom \texttt{WaldoPatchDataset} class is defined to load image patches from directories, resize them, and convert them to tensors.
    \item The dataset is split into training, validation, and test sets, and the distribution of labels is printed.
\end{itemize}

\subsection*{Data Processing}
\begin{itemize}
    \item A function \texttt{compute\_mean\_std} calculates the mean and standard deviation of the RGB channels in the training dataset for normalization purposes.
\end{itemize}

\subsection*{Augmented and Triplet Datasets}
\begin{itemize}
    \item An \texttt{AugmentedDataset} class is defined to handle data augmentation selectively applied to positive samples to address class imbalance.
    \item A \texttt{TripletWaldoDataset} class is created to prepare triplets (anchor, positive, and negative samples) for training with Triplet Loss.
    \item Normalization and augmentation transforms are defined and applied to the datasets.
    \item Data loaders for the triplet datasets are created.
\end{itemize}

\subsection*{Vision Transformer Implementation}
\begin{itemize}
    \item The Vision Transformer model is implemented with classes for patchification (\texttt{Patchify}), feedforward neural network (\texttt{FeedForward}), multi-head attention (\texttt{MultiHeadAttention}), and the main transformer (\texttt{Transformer}).
    \item The \texttt{ViT} class integrates these components and includes a classifier.
\end{itemize}

\subsection*{Triplet Loss Implementation}
\begin{itemize}
    \item The \texttt{TripletLoss} class defines the triplet loss function used to train the model by minimizing the distance between similar images and maximizing the distance between dissimilar images.
\end{itemize}

\subsection*{Training and Evaluation}
\begin{itemize}
    \item Functions \texttt{train\_triplet\_epoch} and \texttt{evaluate\_triplet} are defined to handle training and evaluation of the model using the triplet loss.
    \item The model is trained for a specified number of epochs, with loss and distance metrics being recorded.
    \item The best model state based on validation loss is saved.
\end{itemize}

\subsection*{Results Visualization and Model Saving}
\begin{itemize}
    \item Training and test distances, as well as loss values, are plotted and saved as images.
    \item The final model state is saved to a file for future use.
\end{itemize}

\section*{Key Components Explained}

\subsection*{Custom Dataset and DataLoader}
\begin{itemize}
    \item \texttt{WaldoPatchDataset} loads images and labels, resizes them, and converts them to tensors.
    \item \texttt{AugmentedDataset} and \texttt{TripletWaldoDataset} handle data augmentation and creation of triplets for training with triplet loss.
\end{itemize}

\subsection*{Normalization and Augmentation}
\begin{itemize}
    \item Data normalization ensures consistent input values across the dataset.
    \item Augmentation helps to mitigate class imbalance by creating variations of positive samples.
\end{itemize}

\subsection*{Vision Transformer (ViT)}
\begin{itemize}
    \item \texttt{Patchify}: Divides images into patches and projects them into an embedding space.
    \item \texttt{FeedForward} and \texttt{MultiHeadAttention}: Core components of the transformer model handling transformations and attention mechanisms.
    \item \texttt{Transformer}: Sequentially applies multiple layers of attention and feedforward networks.
    \item \texttt{ViT}: Combines all components and includes a classifier for final output.
\end{itemize}

\subsection*{Triplet Loss}
\begin{itemize}
    \item Used to train the model by ensuring that the distance between an anchor and positive sample is less than the distance between an anchor and negative sample.
\end{itemize}

\subsection*{Training and Evaluation}
\begin{itemize}
    \item Training involves minimizing the triplet loss over epochs, adjusting the model parameters.
    \item Evaluation checks the model's performance on validation and test sets, focusing on the mean distances between positive and negative samples.
\end{itemize}

\subsection*{Visualization}
\begin{itemize}
    \item Results are visualized to understand the model's learning progress and to ensure it is not overfitting.
\end{itemize}
