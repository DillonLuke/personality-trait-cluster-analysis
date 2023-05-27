import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

def correlation_heatmap(data: pd.DataFrame, **kwargs):
    """
    Create a heatmap of correlations between features in data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing features for which correlations are displayed
        in a heatmap
    **kwargs 
        Any keyword arguments to be passed to the sns.heatmap function
    
    """
    corr_mat = data.corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(data=corr_mat, ax=ax, **kwargs)

        
def get_pca_explained_variance(pca_estimator) -> pd.DataFrame:
    """
    Get individual and cumulative variance explained by pca components.
    
    Parameters
    ----------
    pca_estimator : sklearn.decomposition._pca.PCA
        Fitted PCA estimator from which to obtain the individual and 
        cumulative variance explained by PCA component.
    
    Returns
    -------
    df: pd.DataFrame
        Dataframe with individual variance and total (cumulative) variance
        by PCA component. 
    
    """
    results = {
        "Individual Variance": pca_estimator.explained_variance_,
        "Total Variance (%)": pca_estimator.explained_variance_ratio_.cumsum(),
    }
    
    components = np.arange(pca_estimator.n_components_) + 1
    
    df = pd.DataFrame(results, index=pd.Index(components, name="Component Number"))
    
    return df                          


def get_pca_component_values(pca_estimator) -> pd.DataFrame:
    """
    Get principal components in original feature space.
    
    Parameters
    ----------
    pca_estimator : sklearn.decomposition._pca.PCA
        Fitted PCA estimator from which to obtain the principal components in
        original feature space.
    
    Returns
    -------
    df: pd.DataFrame
        Dataframe with dimensions of each principal component in the original 
        feature space. Column names reflect original feature names, and index
        reflects component number.
    
    """
    df = pd.DataFrame(
        data=pca_estimator.components_,
        index=[f"Component {i+1}" for i in range(pca_estimator.n_components_)],
        columns=pca_estimator.feature_names_in_
    )
    
    return df 


def dual_axis_lineplot(df: pd.DataFrame, y1: str, y2: str, **kwargs):
    """
    Plot two lineplots with different y-value axes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns containing both sets of y-values.
    y1 : str
        Name of the column containing first set of y-values.
    y2 : str
        Name of the column containing second set of y-values.
    **kwargs
        Any keyword arguments to be passed to the pd.DataFrame.plot method.
    
    """
    y_cols = [y1, y2]
    
    ax = df[y_cols].plot(secondary_y=[y2], **kwargs)
    
    axes = [ax, ax.right_ax]
    for col, axis in zip(y_cols, axes):
        line = axis.get_lines()[0]
        axis.set_ylabel(col, color=line.get_color())
    
    
def kmean_gridsearch(df: pd.DataFrame, ks, **kwargs):
    """
    Perform gridsearch for k-means clustering over a range of k values.
    
    This function will perform a gridsearch for k-means clustering over
    a user-specified range of k values and record the inertia and silhouette
    score corresponding to each k value.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data that will be used to fit the KMeans
        estimators for each k value.
    ks : array-like
        Array containing the k values over which to fit KMeans estimators
        and record inertia and silhouette scores.
    
    Returns
    -------
    km_ests : list
        List of sklearn KMeans estimators fitted over the 
        user-specified range of k values.
    km_df: pd.DataFrame
        Dataframe which includes (1) k values, (2) inertia scores, and
        (3) silhouette scores. The index reflects k values.
    
    """
    km_ests = []
    km_results = []
    for k in ks:
        km_ests.append(KMeans(n_clusters=k, **kwargs).fit(df))
        
        km_result = {
            "k": k,
            "inertia": km_ests[-1].inertia_,
            "silhouette": silhouette_score(df, km_ests[-1].labels_)
        }
    
        km_results.append(km_result)
    
    km_df = pd.DataFrame(km_results).set_index("k")
    
    return km_ests, km_df


def gmm_gridsearch(df: pd.DataFrame, cluster_nums, **kwargs):
    """
    Perform gridsearch for Gaussian Mixture Models over a range of cluster numbers.
    
    This function will perform a gridsearch for Gaussian Mixture Models over
    a user-specified range of cluster numbers and record the AIC and BIC 
    scores corresponding to each number of clusters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data that will be used to fit the 
        GaussianMixture estimator for each cluster number value.
    cluster_nums : array-like
        Array containing the cluster number values over which to fit 
        GaussianMixture estimators and record AIC and BIC scores.
    
    Returns
    -------
    gmm_ests : list
        List of sklearn GaussianMixture estimators fitted over the 
        user-specified range of cluster numbers.
    gmm_df: pd.DataFrame
        Dataframe which includes (1) cluster number, (2) AIC scores, and
        (3) BIC scores. The index reflects cluster number.
    
    """
    gmm_ests = []
    gmm_results = []
    for num in cluster_nums:
        gmm_ests.append(GaussianMixture(n_components=num, **kwargs).fit(df))
        
        gmm_result = {
            "Cluster Number": num,
            "aic": gmm_ests[-1].aic(df),
            "bic": gmm_ests[-1].bic(df)
        }
    
        gmm_results.append(gmm_result)
    
    gmm_df = pd.DataFrame(gmm_results).set_index("Cluster Number")
    
    return gmm_ests, gmm_df
    

def cluster_animation(df: pd.DataFrame, cluster_estimators, cluster_nums, **kwargs):
    """
    Animate cluster solutions by number of clusters in an interactive 3D scatterplot.
    
    This function provides an animation of a user-specified range of different cluster
    solutions. Each cluster solution constitutes one frame and is a scatterplot with
    the original data plotted in 3D feature space colored according to their assigned
    cluster.
    
    Parameters
    ----------
    df : pd.DataFrame of shape (n, 3)
        Dataframe on which the cluster estimators were fit. This dataframe
        should include 3 columns representing the original 3D feature space.
    cluster_estimators : array-like of shape (n_estimators)
        Array of estimators used to cluster data in df. The cluster estimator
        should have a predict method.
    cluster_nums : array-like of shape (n_estimators)
        Array of cluster numbers corresponding to cluster estimators. This 
        array should be the same length as cluster_estimators.
    **kwargs
        Any keyword arguments to be passed to the px.scatter_3d function.
    
    """
    cluster_pred_df = pd.DataFrame(
        data=np.transpose([est.predict(df) for est in cluster_estimators]),
        columns=cluster_nums,
        index=df.index
    )
    
    df_combined = pd.concat((df, cluster_pred_df), axis=1)
    
    df_animation = df_combined.melt(
        id_vars = df.columns.tolist(),
        var_name="Cluster Number",
        value_name="Label",
        ignore_index=False
    )
    
    return px.scatter_3d(data_frame=df_animation, x=df.columns[0], y=df.columns[1],
                         z=df.columns[2], color="Label", animation_frame="Cluster Number",
                         **kwargs)
