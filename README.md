# BRAQUE: Bayesian Reduction for Amplified Quantization in UMAP Embedding


## Lazy users Guide

Just download and install the requirements with `pip install -r requirements.txt`, then install braque with `pip install -i https://test.pypi.org/simple/ braque`, we suggest to do so in an empty enviroment.
Then download the examples folder at https://github.com/LorenzoDallOlio/BRAQUE/tree/main/examples, and see if you can run BRAQUE-usage.py (it should take around a minute the whole procedure.

Then replace the mòandatory things with your tidy data (original_db.csv, positions.csv, reference.csv, and edit the strings in the BRAQUE-usage.py accordingly).

The basic usage is passing every argument to the class constructor, and then simply run ___self.run_pipeline()___ from the constructed object, or single parts of the pipeline if you want so.

If you wish there is also a python notebook version of BRAQUE-usage. For further boring details check the following sections

You can find the BRAQUE paper here for further details: https://doi.org/10.3390/e25020354
If you use part of BRAQUE code please cite our paper, together with the 3 main references, depending on which part of the pipeline you used.

To understand BRAQUE's class check the [BRAQUE-standard](#BRAQUE-standard) section.
To cherry-pick functions from BRAQUE and reproduce our paper results check the [BRAQUE-legacy](#BRAQUE-legacy) section.

<br><br>


## General Introduction

BRAQUE is a pipeline for clustering high dimensional data, specifically it was built for and tested on proteomics data acquired through immunofluorescence technique.

In brief, BRAQUE uses a new preprocessing techinque, called Lognormal Shrinkage, to estimate candidate subpopulations within each marker distribution. 
Then these candidate subpopulations are separated, embedded with the UMAP[^1] algorithm, and clustered with one or two iterations of the HDBSCAN[^2] algorithm. 
Lastly, a robust measure of effect size[^3] is computed to estimate which markers are more characteristic for every cluster, and many summarizing plots can be performed.


<br><br>

## Installation


Download the requirements.txt file from Github (https://github.com/LorenzoDallOlio/BRAQUE/).
To download and use braque we suggest making it in a separate enviroment, install pip and then run

```
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ braque
```
to download the latest version of BRAQUE

<br><br>
<br><br>


## BRAQUE-standard

### how to use BRAQUE-standard?

The idea behind this section is to provide you with a deep explanation of the BRAQUE class and the easy-to-use examples.

This section explains the python class installed and imported with standard procedure, and shows an example of usage for such class, which can be found within the examples directory on github (https://github.com/LorenzoDallOlio/BRAQUE/tree/main/examples) with the name BRAQUE-usage (you can find both a notebook and a python script) together with 3 toy dataset needed to run the examples (original_db-toy.csv, postions-toy.csv, and reference-toy.csv). Just download them, install braque, place them in a directory and run the script/notebook to check BRAQUE results/inputs/plots/etc.




### The BRAQUE class

___\_\_init\_\_(self, original_db, pos, reference, correspondence_column, naming_column, interpretative_column, importance_column, features_subgroup='', area='dataset', perform_features_selection=True, perform_lognormal_shrinkage=True, subsampling=1, max_n_gaussians=15, contraction_factor=5., populations_plot=True, nn=50, metric='euclidean', HDBSCAN_merging_parameter=0.1, reclustering_step=False, p_val_basic_threshold = 0.05, load_embed=False, load_db=False, load_clusters=False, base_folder='./', save_plots=True, verbose=False)___:


Class constructor for BRAQUE object to perform the pipeline reported in Dall'Olio et al. https://doi.org/10.3390/e25020354.  

1. __original_db__ _(pandas DataFrame, n_cells x n_features shaped)_: data on which to perform the analysis, with units (e.g., cells) on rows and features (e.g. features) on columns. This variable will remain untouched, and will be used at the end for statistical comparisons.
2. __pos__ _(pandas DataFrame, n_cells x 2 shaped)_: spatial positional features for your units (e.g. x,y columns for real space coordinates).
3. __reference__ _(pandas DataFrame, n_features x n_properties shaped)_: DataFrame where every row must correspond to a different feature and every column should provide a different property of such feature, few columns are mandatory, like corresponding_column, naming_column, interpretative_column and importance_column, see below for further details.
4. __correspondence_column__ _(str)_: header of the reference file column which contains features names, may contain multiple variants in the format "variant1/variant2/.../variantN".
5. __naming_column__ _(str)_: header of the reference file column which contains features names that shall be used in plots/results.
6. __interpretative_column__ _(str)_: header of the reference file column which contains features associated property.
7. __importance_column__ _(str)_: header of the reference file column which contains 1 for important features that should be used for summary plot.
8. __features_subgroup__ _(str)_: optional header of the reference file column which might opionally be used to keep only a subset of features. if used shall be a 0/1 coded column, with 1 for keeping the feature at that specific row, or 0 to exlude it. use an empty string ("") to avoid such subselection.
9. __area__ _(str)_: string that will be used for naming the folders and plots correspond to the current dataset/analysis.
10. __perform_feature_selection__ _(boolean)_: whether or not to perform features selection.
11. __perform_lognormal_shrinkage__ _(boolean)_: whether or not to perform lognormal shrinkage (preprocessing from Dall'Olio et al.).
12. __subsampling__ _(integer, between 1 and len(db))_: subsampling parameter, take 1 cell every N. in order to speed up gaussian mixture fitting procedure
13. __max_n_gaussians__ _(positive integer, >=2)_: maximum number of fittable lognormal distributions for a single feature, keep in mind that the higher the slower and more precise the algorithm. To tune follow guidelines from Dall'Olio et al.
14. __contraction_factor__ _(positive float, >1.)_: each gaussian in the log2 space is contracted by this factor to better separate candidate subpopulations. To tune follow guidelines from Dall'Olio et al.
15. __populations_plot__ _(boolean)_: whether or not to plot the final summary about number of candidates subpopulations for each feature, useful to tune max_n_gaussians.
16. __nn__ _(integer)_: number of nearest neighbors to use during UMAP
17. __metric__ _(str, one of scipy-allowed distances)_: which metric to use during UMAP algorithm
18. __HDBSCAN_merging_parameter__ _(float, non-negative)_: corresponds to 'cluster_selection_epsilon' of the HDBSCAN algorithm.
19. __reclustering_step__ _(boolean)_: whether or not to perform a second HDBSCAN clustering on the biggest cluster to unpack eventual superclusters that may form in immunofluorescence context, do not use if you are not sure.
20. __p_val_basic_threshold__ _(float, between 0 and 1 excluded)_: which interpretative_column threshold should be adopted for a single test, such threshold will be bonferroni corrected for multiple tests scenarios.
21. __load_embed__ _(boolean)_: whether or not to load precomputed embedding from the /embeddings/ subfolder.
22. __load_db__ _(boolean)_: whether or not to load precomputed processed db from the /quantized_dbs/ subfolder. 
23. __load_clusters__ _(boolean)_: whether or not to load precomputed clusters from the /results/area/ subfolder.
24. __base_folder__ _(str)_: root folder from which the analysis tree will start and be performed, within this folder plots and results will e stored in appropriate subfolders.
25. __save_plots__ _(boolean)_: whether or not to store the produced plots.
26. __verbose__ _(boolean)_: whether or not to obtain a more verbose output.
</br></br></br>






___custom_colormap_for_many_clusters(self, n_clusters=None, random_seed=42, bright_threshold=0.2)___:

New colormap to deal properly with 20+ clusters scenarios.

1. __n_clusters__ _(integer)_: number of clusters, each of which will correspond to a color in the resulting output
2. __random_seed__ _(integer)_: random seed for color order, different seeds will give different color orders
3. bright_threshold__ _(float, between 0 an 1)_: value used to discard shades of white and very bright colors, the higher the less colors will be used for the colormap

</br></br></br>





___find_names(self, features_names)___:

Substitutes columns names with pre-defined standard names contained in reference file.

1. __features_names__ _(string or list/array-like)_: either single string to convert or list of strings to convert to standard name. It is important for these values to exactly correspond to a value of the reference file corresponding column.



</br></br></br>




___add_main_features_interpretative_columns(self, find_n=3, undef_thr=0.)__:

Add inplace to input pandas dataframe the column 'MainFeatures' and a column with their interpretation.

1. __find_n__ _(positive integer)_: how many main features to find, at most
2. __undef_thr__ _(non-negative float): threshold below which an effect size is never considered relevant

</br></br></br>






___features_selection(self, drop_unclear=True, drop_missing=True, to_drop = ['IGNORE'], special_keeps=[])___:

Perform features selection over a dataframe, given a reference file on which column to keep/discard.        

1. __drop_unclear__ _(boolean)_: whether to drop features with no corrispondence in the reference file
2. __drop_missing__ _(boolean)_: whether to drop features with missing interpretative_column
3. __to_drop__ _(list or array-like)_: drop features whose column named 'interpretative_column' value, in the reference file, is part of this list
4. __special_keeps__ _(list or array-like)_: features whose name is in this lisy will be kept anyway if they have at least a interpretative_column and it's != IGNORE



</br></br></br>




___lognormal_shrinkage(self, subsampling=1, max_n_gaussians=20, log_transform=True, contraction_factor=5., populations_plot=False)___:

Perform Lognormal Shrinkage preprocessing over a pandas datafame.

1. __subsampling__ _(positive integer, between 1 and len(db))_: subsampling parameter, take 1 cell every N. in order to speed up gaussian mixture fitting procedure
2. __max_n_gaussians__ _(positive integer, >=2)_: maximum number of fittable lognormal distributions for a single feature, keep in mind that the higher the slower and more precise the algorithm. To tune follow guidelines from Dall'Olio et al.
3. __log_transform__ _(boolean)_: whether to do a lognormal mixture (suggested if data are not very multi gaussian) or a gaussian mixture.
4. __contraction_factor__ __(positive float, >1.)_: each gaussian in the log2 space is contracted by this factor to better separate candidate subpopulations. To tune follow guidelines from Dall'Olio et al.
5. __populations_plot__ _(boolean)_: whether or not to plot the final summary about number of candidates subpopulations for each feature, useful to tune max_n_gaussians.


</br></br></br>






___embed_dataset(self, nn=50, metric='euclidean', save_embed=True)___:

Perform the embedding on a 2D manifold of a pandas dataframe using the UMAP algorithm.

1. __nn__ _(integer)_: number of nearest neighbors to use during UMAP
2. __metric__ _(str, one of scipy-allowed distances)_: which metric to use during UMAP algorithm
3. __save_embed__ _(boolean)_: whether or not to save the resulting coordinates in the embedding space.


</br></br></br>






___features_importance_per_cluster(self, select='original', compare_with='rest')___:

Compute features importance within each cluster.

1. __select__ _(str)_: select which dataframe to use for effect size and p-value computations, either original robustly scaled ('original', suggested) or preprocessed db (any other string).
2. __compare_with__ _(str, either 'rest' or 'all')_: whether or not to compare each cluster with the rest of the cells or with all the cells (including the cluster itself), the second option could be used to have a common reference for effect sizes and compare different features effect sizes, but qould make the two samples for the ttest not indipendent.





</br></br></br>



___most_important_cluster_effect_sizes(self, rs, rs_p, n=10, cluster_number=-2, path="./cluster__expression.png")___:

Horizontal bar plot with highest effect size features for a given cluster.

1. __rs__ _(numpy array)_: a row for the 'res' output by 'features_importance_per_cluster', represents each feature's effect size for a given cluster
2. __rs_p__ _(numpy array)_: a row for the 'res_p' output by 'features_importance_per_cluster', represents each feature's p-value for a given cluster
3. __n__ _(integer)_: how many features to plot, in descending order of effect size
4. __cluster_number__ _(integer, non-negative)_: number to use as header for cluster name, useful to distinguish multiple clusters.
5. __path__ _(str)_: where to store the resulting plot (if saved)


</br></br></br>






___plot_features_on_embedding(self, basic_path="./features/")___:

Summary plot for all features.

1. __basic_path__ _(str)_: path at which resulting plots should be stored, if saved



</br></br></br>





___plot_embedding_clusters(self, clusterer, path="./__clusters.png")___:

Plot HDBSCAN clusters onto real space and UMAP embedding coordinates with appropriate colormap-

1. __clusterer__ _(fitted hdbscan.HDBSCAN clustering object)_: fitted HDBSCAN clustering object used to cluster data from the embedding space
2. __path__ _(str)_: path where to store the resulting plots if saved



</br></br></br>





___whole_dataset_summary_plots(self, alpha=0.9, legend_size=200, plot_with_legend=True, plot_without_legend=True, plot_noise=True)__:

Plot a Maximum of 4 plots (2 with legend and 2 without legend) which summarize the main features and their self.interpretative_column for each cluster.

1. __alpha__ _(float between 0 and 1)_: transparency for the plots
2. __legend_size__ _(positive float)_: size for the dots in the legend
3. __plot_with_legend__ _(boolean)_: whether or not to make the 2 plots with legends (may be unreadable if too many clusters/too long labels are used)
4. __plot_without_legend__ _(boolean)_: whether or not to make the 2 plots without legends (useful if with legends the plot are unreadable, please notice that this parameter is not mandatory to be opposite to 'plot_with_legend')
5. __plot_noise__ _(boolean)_: whether or not to plot the noise cluster if found by HDBSCAN




</br></br></br>




___find_monitoring_features(self)___:
        
Find which features are the main ones accoring to self.importance_column.



</br></br></br>





___plot_cluster_spatial_location(self, tiers_label, mask, cluster_number=-2, path="./cluster__position.png")___:

Plot cluster position in UMAP embedding and in real space, together with a summary of Tier1 and Tier2 features.

1. __tiers_label__ _(str)_: string produced by func containing Tier 1 and Tier 2 features, which will be used as header for the plot
2. __mask__ _(array-like of boolean values)_: boolean values corresponding to whether or not a single cell is part of the considered cluster
3. __cluster_number__ _(integer, non-negative)_: number to use as header for cluster name, useful to distinguish multiple clusters.
4. __path__ _(str)_: where to store the resulting plot (if saved).



</br></br></br>





___monitoring_features_plot(self, mask, cluster_number=-2, palette=[(0.2, 0.5, 1.), (1., 0.2, 0.2)], path='./cluster__monitoring_featuresining.png')___:

Plot Kernel Density Estimation to compare a cluster with whole dataset using some monitoring features.

1. __mask__ _(array-like of boolean values)_: boolean values corresponding to whether or not a single cell is part of the considered cluster
2. __cluster_number__ _(integer, non-negative)_: number to use as header for cluster name, useful to distinguish multiple clusters.
3. __monitoring_features__ _(list or array-like)_: list of features over which each cluster will be compared with whole sample using KDE plots
4. __palette__ _(list of 2 rgb tuples)_: color palette to use for cluster or whole sample cells and KDE plots
5. __path__ _(str)_: where to store the resulting plot (if saved).

 



</br></br></br>



___run_pipeline(self)___:

Run the complete BRAQUE pipeline from preprocessing to plots and storing results.



</br></br></br>



### BRAQUE usage

From the examples folder on github (https://github.com/LorenzoDallOlio/BRAQUE/tree/main/examples) download the script and the .csv files. Then you should have everything ready after braque installation, just try to run the script.

The basic usage is to call the class constructor with the desired parameters and datasets, and lastly call the ___self.run_pipeline()___ method of the built object. Then just by substituting the dataset with your own everything should be ready, please notice file formats and shapes to be coherent with toy datasets, as well as substituing with the new correct headers the parameters ___self.interpretative_column___, ___self.correspondence_column___, ___self.naming_column___, and ___self.importance_column___ at least.


</br></br></br>
</br></br></br>




## BRAQUE-legacy

### How to use BRAQUE-legacy?


The idea of this section is explaining you the functions and how to use them, in order to let you combine them as you prefer if you want to try BRAQUE or some parts of it.

within the examples directory on github (https://github.com/LorenzoDallOlio/BRAQUE/tree/main/examples) you can find both a notebook and a python script named BRAQUE-legacy. This section explains them.

This version is provided in order to reproduce the results obtained in the published the paper over the publicly available datasets we provided (http://dx.doi.org/10.17632/j8xbwb93x9.1), and therefore they both consist of a code which is ready to run on a specific kind of dataset, and should/could be adjusted to your needs for different datasets.

To prepare everthing you can install the dependencies in the requirements.txt file, but feel free to change versions as long as the algorithm runs till the end it should have worked properly.
You can use _'conda install requirements.txt'_ or _'pip install -r requirements.txt'_ for this task.



#### Input Data

The essential elements to properly run BRAQUE are the following variables:

1. __original_db__: your data stored in a .csv file, with units (e.g., cells) on rows and features (e.g. markers) on columns. This variable will remain untouched, will be used at the end for statistical comparisons
1. __db__: a copy of original_db, which will undergo the pipeline, overwriting itself after features selection and preprocessing
2. __pos__: positional features for your units (e.g. x,y columns for real space coordinates)
3. __reference__: a reference file, in .csv format, where every row must correspond to different markers. This reference file needs to have few more columns to fully perform the whole pipeline:
    1. __correspondence_column__: header of the reference file column which contains features names, may contain multiple variants in the format "variant1/variant2/.../variantN"
    2. __naming_column__: header of the reference file column which contains features names that shall be used in plots/results
    3. __interpretative_column__: header of the reference file column which contains features associated property
    4. __importance_column__: header of the reference file column which contains 1 for important features that should be used for summary plot
    5. _(optional)_ __markers_subgroup__: header of the reference file column which might opionally be used to keep only a subset of features. if used shall be a 0/1 coded column, with 1 for keeping the feature at that specific row, or 0 to exlude it. use an empty string ("") to avoid such subselection
    
Please notice that in our paper specific code we used original_db with 3 more columns, that were x, y, and area of each cell. Therefore we obtain db from selecting the first N-3 columns, and pos from the columns N-2 and N-1, discarding the area. Feel free to prepare the data in a less intricated way...





#### Setup script parameters

At the begin of the script there are some further parameters you can play with. This organization will be moved in a class constructor, but as previously mentioned the paper specific code is maintained as untouched as possible to reproduce paper results. The important parameters to set are:
1. __output_file__: string, file path where to store printed outputs in a text file
2. __verbose_output__: boolean, whether or not to produce a more verbose output
3. __area__: string, file path from which extract the original_db dataframe
4. __reference_file__: string, file path from which to extract to reference .csv file
5. __correspondence_column__: string, header of the reference file column which contains features names, may contain variants
6. __naming_column__: string, header of the reference file column which contains features names that shall be used in plots/results
7. __interpretative_column__: string, header of the reference file column which contains features associated propery
8. __importance_column__: string, header of the reference file column which contains 1 for important features that should be used for summary plot
9. __markers_subgroup__: string, header of the reference file column which might opionally be used to keep only a subset of features. if used shall be a 0/1 coded column, with 1 for keeping the feature at that specific row, or 0 to exlude it. use an empty string ("") to avoid such subselection
10. __perform_features_selection__: boolean, whether or not to perform features selection over db using reference file
11. __perform_lognormal_shrinkage__: boolean, wether or not perform LNS (Lognormal Shrinkage preprocessing from Dall'Olio et al. https://doi.org/10.3390/e25020354
12. __max_n_gaussians__: integer, maximum number of subpopulation that can be identified in a single feature, the higher the number the more precise and slow the analysis will be
13. __contraction_factor__: float >1, separation parameter for candidate subpopulations
14. __nn__: integer, UMAP number of nearest neighbors to use
15. __metric__: string, metric to use within UMAP algorithm 
16. __load_embed__: boolean, whether or not to load the premade embedding obtained from UMAP
17. __load_db__: boolean, whether or not to load the premade preprocessed db
18. __load_clusters__: boolean, whether or not to load the premade clusters
19. __large_db_procedure__: boolean, whether or not to use large db procedure, which consists of loading just the embedding, clustering it, and finally loading the remaining dbs, use this if clustering does not work for memory issues

All global parameters are provided with a default value, for suggestion on values or ranges for the main pipeline parameters please consult the table 1 of our paper and the relative discussion.





#### Functions


1. Utility functions:
    1. __custom_colormap_for_many_clusters__ _(n_clusters=None, random_seed=42, bright_threshold=0.2)_: New colormap to deal properly with 20+ clusters scenarios.
    
    2. __find_names__ _(markers_names)_: Substitutes columns names with pre-defined standard names contained in reference file.
    
    3. __add_main_markers_significance_columns__ _(db, res, res_p, find_n=3, undef_thr=0., thrs_p=0.05)_: Add inplace to input pandas dataframe the column 'MainMarkers' and a column with their interpretation.


2. Main pipeline functions:
    1. __features_selection__ _(db, reference, drop_unclear=True, drop_missing=True, to_drop = \['\IGNORE'\], special_keeps=\[\])_: Perform features selection over a dataframe, given a reference file on which column to keep/discard.
    
    2. __lognormal_shrinkage__ _(db, subsampling=1, max_n_gaussians=20, contraction_factor=5., populations_plot=False)_: Perform Lognormal Shrinkage preprocessing over a pandas datafame.
    
    3. __embed_dataset__ _(db, nn=50, metric='euclidean', save_embed=True)_: Perform the embedding on a 2D manifold of a pandas dataframe using the UMAP algorithm.
    
    4. __markers_importance_per_cluster__ _(use, clusters, compare_with='rest')_: Compute markers importance within each cluster.
    
    5. __most_important_effect_sizes__ _(rs, rs_p, n, thrs_p=0.05, save_plot=False, path="./cluster__expression.png")_: Horizontal bar plot with highest effect size markers for a given cluster.


3. Plots related functions:
    1. __plot_markers_on_embedding__ _(db, original_db, embedding, real_pos, save=False, path=folder+"markers/")_: Summary plot for all markers.
    
    2. __plot_embedding_clusters__ _(clusters_list, real_pos, embedding, clusterer, save_plot=False, path="./\_\_clusters.png")_: Plot HDBSCAN clusters onto real space and UMAP embedding coordinates with appropriate colormap.
    
    3. __whole_dataset_summary_plots__ _(db, area, alpha=0.9, size=1., legend_size=200, plot_with_legend=True, plot_without_legend=True, plot_noise=True, save_plot=False, base_path="./")_: Plot a Maximum of 4 plots (2 with legend and 2 without legend) which summarize the main markers and their interpretative_column for each cluster.
    
    4. __plot_cluster_spatial_location__ _(lbl, embedding, pos, mask, save_plot=False, path="./cluster__position.png")_: Plot cluster position in UMAP embedding and in real space, together with a summary of Tier1 and Tier2 markers.
    
    5. __monitoring_features_plot__ _(original_db, mask, cluster_name, monitoring_features, palette=[(0.2, 0.5, 1.), (1., 0.2, 0.2)], save_plot=False, path='./cluster__monitoring_featuresining.png')_: Plot Kernel Density Estimation to compare a cluster with whole dataset using some monitoring features.



Please notice that the tag ___\<specific for the paper\>___ is referred to portions of code that are useless for a general dataset, but needed for our specific dataset to reproduce the paper's results.



#### Pipeline order


The main pipeline can be summarized as follows:

1. Initialize the data
2. Run Features selection (using __features_selection__ function)
3. Run Lognormal Shrinkage preprocessing (using __lognormal_shrinkage__ function)
4. Run dimensionality reduction with UMAP (using __embed_dataset__ function)
5. Run clustering on the UMAP embedding with HDBSCAN (using the __hdbscan.HDBSCAN()__ clusterer and running its fit and predict methods)
6. Compute the robust effect size index for every marker/cluster combination (using __markers_importance_per_cluster__ function)
7. Plot the desired results with appropriate functions, chosen from the Plots subsection of the "Function" section


All plots and partial results to load are stored by default in repositories created within the working directory.


<br><br>


## Contacts

For questions regarding BRAQUE feel free to contact Lorenzo Dall'Olio at "lorenzo.dallolio4@unibo.it" or "lorenzo.dallolio@studio.unibo.it"

<br><br>


## References

1. Dimensionality Reduction:
[^1]: McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018

2. Clustering:
[^2]: Campello, R.J.G.B., Moulavi, D., Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. In: Pei, J., Tseng, V.S., Cao, L., Motoda, H., Xu, G. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2013. Lecture Notes in Computer Science(), vol 7819. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-37456-2_14

3. Robust effect size index:
[^3]: Vandekar, S., Tao, R., & Blume, J. (2020). A Robust Effect Size Index. Psychometrika, 85(1), 232–246. https://doi.org/10.1007/s11336-020-09698-2
