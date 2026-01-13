
import os
import streamlit as st

from rl_inventory.scripts.evaluate import evaluate_pretrained_all

st.set_page_config(page_title="Inventory RL Demo", layout="wide")
st.title("Inventory Management RL — Pretrained Demo")

st.markdown("This demo loads pretrained agents from the repository and evaluates them.")

num_episodes = st.slider("Evaluation episodes", 1, 30, 10)
base_seed = st.number_input("Base seed", value=0, step=1)

run = st.button("Run evaluation (pretrained)", type="primary")

if run:
    with st.spinner("Loading models and evaluating…"):
        out = evaluate_pretrained_all(num_episodes=int(num_episodes), base_seed=int(base_seed))

    st.success("Done.")
    st.dataframe(out["df"], use_container_width=True)

    if os.path.exists(out["plot_path"]):
        st.image(out["plot_path"], caption="Comparison plot")

    st.download_button("Download CSV", data=open(out["csv_path"], "rb"), file_name="results_summary.csv")
    st.download_button("Download PDF report", data=open(out["pdf_path"], "rb"), file_name="evaluation_report.pdf")

    with st.expander("Console output"):
        st.text(out["captured_output"])
