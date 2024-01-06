import plotly.express as px
import pandas as pd
import  streamlit as st

def build_model_PCA(df):
    from sklearn.decomposition import PCA

    fig=px.scatter_matrix(df, dimensions=ind_var, color=target_var)
    st.subheader("Pairplot (each feature against other features)")
    st.plotly_chart(fig)

    features = df[ind_var]
    pca = PCA(n_components=n_components, random_state=random_state)
    projections = pca.fit_transform(features)

    if n_components == 2:
        st.subheader("PCA 2D Projection")
        fig=px.scatter(
            projections, x=0, y=1,
            color=df[target_var], labels={'color': target_var}
        )
    else:
        st.subheader("PCA 3D Projection")
        fig=px.scatter_3d(
            projections, x=0, y=1, z=2,
            color=df[target_var], labels={'color': target_var}
        )
        fig.update_traces(marker_size=4)

    st.plotly_chart(fig)

def build_model_TSNE(df):
    from sklearn.manifold import TSNE

    fig=px.scatter_matrix(df, dimensions=ind_var, color=target_var)
    st.subheader("Pairplot (each feature against other features)")
    st.plotly_chart(fig)

    features=df[ind_var]
    tsne = TSNE(n_components=n_components, random_state=random_state)
    projections = tsne.fit_transform(features)

    if n_components == 2:
        st.subheader("t-SNE 2D Projection")
        fig=px.scatter(
            projections, x=0, y=1,
            color=df[target_var], labels={'color': target_var}
        )
    else:
        st.subheader("t-SNE 3D Projection")
        fig=px.scatter_3d(
            projections, x=0, y=1, z=2,
            color=df[target_var], labels={'color': target_var}
        )
        fig.update_traces(marker_size=4)

    st.plotly_chart(fig)

def build_model_UMAP(df):
    import umap.umap_ as umap

    fig=px.scatter_matrix(df, dimensions=ind_var, color=target_var)
    st.subheader("Pairplot (each feature against other features)")
    st.plotly_chart(fig)

    features=df[ind_var]
    umap_proj = umap.UMAP(n_components=n_components, random_state=random_state)
    projections = umap_proj.fit_transform(features)

    if n_components == 2:
        st.subheader("UMAP 2D Projection")
        fig=px.scatter(
            projections, x=0, y=1,
            color=df[target_var], labels={'color': target_var}
        )
    else:
        st.subheader("UMAP 3D Projection")
        fig=px.scatter_3d(
            projections, x=0, y=1, z=2,
            color=df[target_var], labels={'color': target_var}
        )
        fig.update_traces(marker_size=4)

    st.plotly_chart(fig)


# Page Layout ( Streamlit web Interface )
st.set_page_config(page_title="Dimensionality Reducer")

st.write("""
# Dimensionality Reducer
""")

# Sidebar ..............................................

# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.header("Select Dimension Reducer")
reg = st.sidebar.selectbox("Choose Dimensionality Reduction Algorithm", options=['Principal Component Analysis (PCA)',
                                                                                 't-Distributed Stochastic Neighbor Embedding (t-SNE)'
                                                                                 ])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = px.data.iris()

st.sidebar.subheader('Variable Configuration')
ind_var = st.sidebar.multiselect('Choose Features', options=df.columns)
target_var = st.sidebar.selectbox('Choose Target Variable', options=df.columns)

st.sidebar.write('---')
n_components = st.sidebar.selectbox("2D or 3D", options=['2', '3'])
n_components = int(n_components)

random_state = st.sidebar.number_input("Random State", value=0)


#######################################################################################
st.subheader('Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    if reg == 'Principal Component Analysis (PCA)':
        build_model_PCA(df)
    if reg == 't-Distributed Stochastic Neighbor Embedding (t-SNE)':
        build_model_TSNE(df)
    # if reg == 'Uniform Manifold Approximation & Projection (UMAP)':
    #     build_model_UMAP(df)


else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Iris Dataset'):
        df = px.data.iris()

        st.markdown('The **IRIS** dataset is used as the example.')
        st.write(df.head(5))

        if reg == 'Principal Component Analysis (PCA)':
            build_model_PCA(df)
        if reg == 't-Distributed Stochastic Neighbor Embedding (t-SNE)':
            build_model_TSNE(df)
        # if reg == 'Uniform Manifold Approximation & Projection (UMAP)':
        #     build_model_UMAP(df)