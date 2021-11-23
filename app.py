import streamlit as st
from huggingpics.data import get_image_urls_by_term


def show_images_of_term(search_term, num_cols=5, num_rows=3):

    # Get the image urls
    # Arbitrarily adding 2 to make sure we have enough images in the event of a failed request
    urls = get_image_urls_by_term(search_term, count=(num_rows * num_cols) + 2)

    st.title(search_term)
    for row_id in range(num_rows):
        cols = st.columns(num_cols)
        for col_id in range(num_cols):
            cols[col_id].image(urls[row_id * num_cols + col_id], use_column_width=True)

def explore():
    with st.sidebar:
        term_1 = st.sidebar.text_input('Search Term 1', value='shiba inu')
        term_2 = st.sidebar.text_input('Search Term 2', value='husky')
        term_3 = st.sidebar.text_input('Search Term 3', value='')
        term_4 = st.sidebar.text_input('Search Term 4', value='')
        term_5 = st.sidebar.text_input('Search Term 5', value='')

        terms = [t for t in [term_1, term_2, term_3, term_4, term_5] if t]

    for term in terms:
        show_images_of_term(term)


def create_dataset():
    st.text("# Coming soon...")


def main():
    
    with st.sidebar:
        mode = st.sidebar.selectbox("Mode", ["Explore", "Create Dataset"])
        st.sidebar.markdown("---")

    _ = explore() if mode == "Explore" else create_dataset()



if __name__ == '__main__':
    main()
