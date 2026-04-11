import streamlit as st

st.set_page_config(page_title="Test", layout="wide")

st.title("Test App")

st.write("App is loading successfully!")

uploaded_file = st.file_uploader("Upload any file")

if uploaded_file:
    st.write(f"Uploaded: {uploaded_file.name}")
    st.write(f"Size: {len(uploaded_file.getbuffer())} bytes")

    st.info("💡 **Tip**: For best results, record **face-on** or **down-the-line** with good lighting and full body visible.")

st.markdown("---")
st.caption("Built with MediaPipe + CrewAI • Running on Apple Silicon MacBook Pro")