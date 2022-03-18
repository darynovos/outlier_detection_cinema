```python
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import pdist,squareform
```

# Data


```python
ratings = pd.read_excel('rating.xlsx')
```


```python
ratings.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.048575e+06</td>
      <td>1.048575e+06</td>
      <td>1.048575e+06</td>
      <td>1.048575e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.527086e+03</td>
      <td>8.648988e+03</td>
      <td>3.529272e+00</td>
      <td>1.096036e+09</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.018424e+03</td>
      <td>1.910014e+04</td>
      <td>1.051919e+00</td>
      <td>1.594899e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>5.000000e-01</td>
      <td>8.254999e+08</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.813000e+03</td>
      <td>9.030000e+02</td>
      <td>3.000000e+00</td>
      <td>9.658382e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.540000e+03</td>
      <td>2.143000e+03</td>
      <td>4.000000e+00</td>
      <td>1.099263e+09</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.233000e+03</td>
      <td>4.641000e+03</td>
      <td>4.000000e+00</td>
      <td>1.217407e+09</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.120000e+03</td>
      <td>1.306420e+05</td>
      <td>5.000000e+00</td>
      <td>1.427764e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Inventing features for analysis
N_views = ratings.groupby(ratings.movieId).count().iloc[:,[0]]
average_rating = ratings.groupby(ratings.movieId).mean().iloc[:,[1]]
data_for_analysis = pd.DataFrame({'N_views':N_views.userId, 'Rating':average_rating.rating})


plt.figure(figsize = (10,6))
plt.scatter(data_for_analysis['Rating'],data_for_analysis['N_views'],c ='darkcyan')
plt.title('Scatterplot of films')
plt.xlabel('Rating')
plt.ylabel('Number of views')
```




    Text(0, 0.5, 'Number of views')




![png](output_4_1.png)


# Clustering


```python
C_data = data_for_analysis.copy()
```


```python
#Clustering k-means
X = StandardScaler().fit_transform(data_for_analysis)
X = np.array(X)

#Elbow curve
Sum_of_squared_distances = []

K = range(1,21)
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(X)
    Sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize = (8,5))
plt.plot(K,Sum_of_squared_distances,'o-',color = 'darkcyan')   
plt.title('Elbow curve')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.show()
```


![png](output_7_0.png)



```python
n = 7 #number of clusters

km = KMeans(n_clusters=n, init='k-means++').fit(X)
labels = km.labels_
C_data['k-means'] = labels+1
```


```python
#vizualization

plt.figure(figsize =(10,6))
c = ['purple', 'seagreen','maroon','y','c','salmon','royalblue','r','black','blue']
cl = C_data['k-means'].unique()

for i in sorted(cl):
    ff = C_data[C_data['k-means'] == i]
    plt.scatter(ff['Rating'],ff['N_views'],color = c[i-1],label = 'Cluster '+str(i))
    plt.title('K-means')
    plt.ylabel('Number of views')
    plt.xlabel('Rating')
    plt.legend(loc = 2)
```


![png](output_9_0.png)



```python
# N of elementes ineach Cluster
C_data.groupby(C_data['k-means']).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N_views</th>
      <th>Rating</th>
    </tr>
    <tr>
      <th>k-means</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2717</td>
      <td>2717</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5489</td>
      <td>5489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3799</td>
      <td>3799</td>
    </tr>
    <tr>
      <th>4</th>
      <td>64</td>
      <td>64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>803</td>
      <td>803</td>
    </tr>
    <tr>
      <th>6</th>
      <td>260</td>
      <td>260</td>
    </tr>
    <tr>
      <th>7</th>
      <td>894</td>
      <td>894</td>
    </tr>
  </tbody>
</table>
</div>



# Outlier Detection

## DBSCAN


```python
#DBSCAN

dbscan = DBSCAN(min_samples = 50,eps = 0.5).fit(X)
C_data['dbscan'] = dbscan.labels_
c = ['salmon','darkcyan','purple', 'seagreen','maroon','y','c','royalblue','r','black','blue']


plt.figure(figsize =(10,6))
for i in sorted(C_data['dbscan'].unique()):
    ff = C_data[C_data['dbscan'] == i]
    plt.scatter(ff['Rating'],ff['N_views'],color = c[int(i+1)])
    plt.title('DBSCAN')
    plt.ylabel('Number of views')
    plt.xlabel('Rating')
    plt.legend(('Outliers', 'Cluster 1','Cluster 2'),loc =2)
```


![png](output_13_0.png)



```python
C_data.groupby(C_data['dbscan']).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N_views</th>
      <th>Rating</th>
      <th>k-means</th>
    </tr>
    <tr>
      <th>dbscan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>203</td>
      <td>203</td>
      <td>203</td>
    </tr>
    <tr>
      <th>0</th>
      <td>13823</td>
      <td>13823</td>
      <td>13823</td>
    </tr>
  </tbody>
</table>
</div>



## Knn


```python
Xknn = np.array(X)
k = 50 #N of nearest neighbors
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(Xknn)
distances, indices = nbrs.kneighbors(Xknn)
max_dist =  distances.max(axis = 1)


threshold_rate = 0.03
N = round(len(max_dist) * threshold_rate)

max_dist_sn_max = sorted(max_dist, reverse = True)[:N]
threshold = min(max_dist_sn_max)

outlier = []
for i in distances:
    g = i[np.where( i > threshold)]
    if len(g)>0:
        outlier.append(1)
    else:
        outlier.append(0)
```


```python
#vizualization
plt.figure(figsize =(10,6))
C_data['knn'] = outlier
c = ['purple','darkcyan','salmon','maroon','y','seagreen','royalblue','r','black','blue']

for i in sorted(C_data['knn'].unique()):
    ff = C_data[C_data['knn'] == i]
    plt.scatter(ff['Rating'],ff['N_views'],color = c[int(i+1)])
    plt.title('K-nn')
    plt.ylabel('Number of views')
    plt.xlabel('Rating')
    plt.legend(('Inliers','Outliers'),loc =2)
```


![png](output_17_0.png)



```python
C_data.groupby(C_data['knn']).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N_views</th>
      <th>Rating</th>
      <th>k-means</th>
      <th>dbscan</th>
    </tr>
    <tr>
      <th>knn</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13606</td>
      <td>13606</td>
      <td>13606</td>
      <td>13606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>420</td>
      <td>420</td>
      <td>420</td>
      <td>420</td>
    </tr>
  </tbody>
</table>
</div>



## Outliers


```python
knn_outliers = C_data[C_data['knn']==1].index
dbscan_outliers = C_data[C_data['dbscan']==-1].index
kmean_outliers =  C_data[C_data['k-means'].isin([4,6])].index
```

# Map matrix with Genres


```python
movies_db = pd.read_excel('movies.xlsx').set_index('movieId')
```


```python
#Choose movies with existing rating
movies = movies_db.loc[C_data.index,]
```


```python
# Split Title and Year
movies_split = movies.title.str.split('(', expand = True)
movies_split[1] = movies_split[1].str.replace(')','')

movies['year'] = movies_split[1]
movies['title'] = movies_split[0]
```


```python
#unique genre
genre=[]
for i in list(movies.genres):
    gg = str(i).split('|')
    for j in gg:
        if j not in genre:
            genre.append(j)
genre = genre[:-2]


#map matrix with Genres
def GENRE(genre,list_movies):
    list_genre = []
    for i in list_movies:
        gg = str(i).split('|')
        if genre in gg:
            list_genre.append(1)
        else:
            list_genre.append(0)
    return list_genre


for i in genre:
    movies[str(i)] = GENRE(i,list(movies.genres))
```


```python

```
