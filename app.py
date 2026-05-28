import streamlit as st
import requests
import json
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
API_URL    = "https://8zo6b8uv63.execute-api.us-east-2.amazonaws.com/prod/query"
S3_BUCKET  = "rag-job-copilot-jds"
S3_KEY     = "processed/processed_jobs.json"
AWS_REGION = "us-east-2"

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Job Copilot",
    page_icon="💼",
    layout="wide"
)

# ── Load processed jobs from S3 for analytics ─────────────────────────────────
@st.cache_data(ttl=300)
def load_processed_jobs():
    try:
        s3   = boto3.client("s3", region_name=AWS_REGION)
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        jobs = json.loads(obj["Body"].read().decode("utf-8"))
        return jobs
    except Exception as e:
        st.warning(f"Could not load analytics data: {e}")
        return []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💼 RAG Job Copilot")
    st.markdown("---")
    st.markdown("""
    **Cloud-native job intelligence system**
    - 30 real job descriptions
    - TF-IDF + Cosine Similarity
    - Skill extraction & normalization
    - AWS S3 · Lambda · API Gateway
    """)
    st.markdown("---")
    st.markdown("**Roles covered:**")
    st.markdown("- ⚙️ Data Engineer (10 JDs)")
    st.markdown("- 🔬 Data Scientist (10 JDs)")
    st.markdown("- 📊 Data Analyst (10 JDs)")
    st.markdown("---")
    st.markdown("**Sample queries:**")
    st.code("What skills are needed for a Data Engineer?")
    st.code("What does a Data Scientist need to know?")
    st.code("Skills for a senior Data Analyst?")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Job Copilot", "📊 Analytics Dashboard"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — JOB COPILOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Ask about any data role")
    st.markdown("Get the top in-demand skills extracted from real job postings.")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Query",
            placeholder="e.g. What skills are needed for a Data Engineer?",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("Search", use_container_width=True, type="primary")

    if search_btn and query:
        with st.spinner("Searching job descriptions..."):
            try:
                response = requests.post(
                    API_URL,
                    data=json.dumps({"query": query}),
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                data = response.json()
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
                st.stop()

        if response.status_code != 200:
            st.error(f"API error: {data.get('error', 'Unknown error')}")
            st.stop()

        top_skills = data.get("top_skills", [])
        top_jobs   = data.get("top_jobs", [])

        # Top Skills
        st.markdown("### 🎯 Top Skills Found")
        st.caption(f"Based on top {len(top_jobs)} matching jobs for: **{query}**")

        skill_colors = {
            "python": "#3776AB", "sql": "#E38C00", "spark": "#E25A1C",
            "aws": "#FF9900", "docker": "#2496ED", "kubernetes": "#326CE5",
            "tensorflow": "#FF6F00", "pytorch": "#EE4C2C", "airflow": "#017CEE",
            "dbt": "#FF694A", "snowflake": "#29B5E8", "kafka": "#231F20",
        }

        badges_html = ""
        for skill in top_skills:
            color = skill_colors.get(skill, "#4A4A8A")
            badges_html += f"""<span style="background-color:{color};color:white;
                padding:6px 14px;border-radius:20px;margin:4px;font-size:14px;
                font-weight:500;display:inline-block">{skill}</span>"""
        st.markdown(f"<div style='margin:10px 0 20px 0'>{badges_html}</div>", unsafe_allow_html=True)

        # Skill frequency bar chart
        skill_counts = {s: top_skills.index(s) + 1 for s in top_skills}
        fig = px.bar(
            x=top_skills[:10],
            y=[len(top_skills) - i for i in range(len(top_skills[:10]))],
            labels={"x": "Skill", "y": "Relevance Score"},
            color=top_skills[:10],
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Top 10 Skills by Relevance"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Matching Jobs
        st.markdown("### 📋 Matching Job Postings")

        role_icons = {"Data Engineer": "⚙️", "Data Scientist": "🔬",
                      "Data Analyst": "📊", "Unknown": "💼"}
        seniority_colors = {"Senior": "🔴", "Mid-level": "🟡", "Entry": "🟢"}

        for i, job in enumerate(top_jobs):
            with st.expander(
                f"{role_icons.get(job['role'],'💼')} {job['title']} @ {job['company']} — Score: {job['score']}",
                expanded=(i == 0)
            ):
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Role", job["role"])
                with c2: st.metric("Seniority", f"{seniority_colors.get(job['seniority'],'')} {job['seniority']}")
                with c3: st.metric("Match Score", f"{job['score']:.1%}")
                with c4: st.metric("Location", job["location"] if job["location"] not in [", ", ""] else "Remote")

                st.markdown("**Skills in this posting:**")
                pills = " ".join([f"`{s}`" for s in job["skills"]])
                st.markdown(pills if pills else "*No skills extracted*")

                st.markdown("**Snippet:**")
                st.markdown(f"> {job['snippet']}...")

    elif search_btn and not query:
        st.warning("Please enter a query before searching.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Hiring Demand Analytics")
    st.markdown("Insights from 30 real job descriptions across Data Engineer, Data Scientist, and Data Analyst roles.")

    jobs = load_processed_jobs()

    if not jobs:
        st.info("Run a search first to generate analytics data, or check your AWS credentials.")
        st.stop()

    df = pd.DataFrame(jobs)

    # ── KPI Metrics ───────────────────────────────────────────────────────────
    st.markdown("### 📌 Overview")
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Job Postings", len(df))
    with k2: st.metric("Roles Covered", df["role"].nunique())
    with k3: st.metric("Unique Skills Found", len(set(s for skills in df["skills"] for s in skills)))
    with k4: st.metric("Avg Skills per Job", f"{df['skills'].apply(len).mean():.1f}")

    st.markdown("---")

    # ── Top Skills by Role ────────────────────────────────────────────────────
    st.markdown("### 🏆 Top Skills by Role")

    roles = ["Data Engineer", "Data Scientist", "Data Analyst"]
    cols  = st.columns(3)
    role_skill_data = {}

    for col, role in zip(cols, roles):
        role_jobs   = df[df["role"] == role]
        all_skills  = [s for skills in role_jobs["skills"] for s in skills]
        skill_counts = Counter(all_skills).most_common(10)
        role_skill_data[role] = dict(skill_counts)

        skill_df = pd.DataFrame(skill_counts, columns=["Skill", "Count"])
        fig = px.bar(
            skill_df, x="Count", y="Skill",
            orientation="h",
            title=f"{role}",
            color="Count",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=380, showlegend=False,
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Cross-Role Skill Overlap ──────────────────────────────────────────────
    st.markdown("### 🔄 Cross-Role Skill Overlap")
    st.caption("Skills that appear across multiple roles — these are the most universally in-demand.")

    de_skills = set(role_skill_data.get("Data Engineer", {}).keys())
    ds_skills = set(role_skill_data.get("Data Scientist", {}).keys())
    da_skills = set(role_skill_data.get("Data Analyst", {}).keys())

    all_three  = de_skills & ds_skills & da_skills
    de_ds_only = (de_skills & ds_skills) - da_skills
    de_da_only = (de_skills & da_skills) - ds_skills
    ds_da_only = (ds_skills & da_skills) - de_skills

    oc1, oc2, oc3, oc4 = st.columns(4)
    with oc1:
        st.markdown("**All 3 roles**")
        for s in sorted(all_three):
            st.markdown(f"✅ `{s}`")
    with oc2:
        st.markdown("**DE + DS only**")
        for s in sorted(de_ds_only):
            st.markdown(f"🔵 `{s}`")
    with oc3:
        st.markdown("**DE + DA only**")
        for s in sorted(de_da_only):
            st.markdown(f"🟠 `{s}`")
    with oc4:
        st.markdown("**DS + DA only**")
        for s in sorted(ds_da_only):
            st.markdown(f"🟣 `{s}`")

    st.markdown("---")

    # ── Seniority Distribution ────────────────────────────────────────────────
    st.markdown("### 👥 Seniority Distribution")

    sc1, sc2 = st.columns(2)

    with sc1:
        seniority_counts = df["seniority"].value_counts().reset_index()
        seniority_counts.columns = ["Seniority", "Count"]
        fig = px.pie(
            seniority_counts, values="Count", names="Seniority",
            title="Overall Seniority Mix",
            color_discrete_sequence=["#E74C3C", "#F39C12", "#2ECC71"]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        seniority_role = df.groupby(["role", "seniority"]).size().reset_index(name="Count")
        fig = px.bar(
            seniority_role, x="role", y="Count", color="seniority",
            title="Seniority by Role",
            barmode="group",
            color_discrete_map={"Senior": "#E74C3C", "Mid-level": "#F39C12", "Entry": "#2ECC71"}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Overall Top 20 Skills ─────────────────────────────────────────────────
    st.markdown("### 📈 Overall Top 20 In-Demand Skills")

    all_skills_flat = [s for skills in df["skills"] for s in skills]
    top20 = pd.DataFrame(Counter(all_skills_flat).most_common(20), columns=["Skill", "Count"])

    fig = px.bar(
        top20, x="Skill", y="Count",
        color="Count",
        color_continuous_scale="Viridis",
        title="Top 20 Skills Across All 30 Job Descriptions"
    )
    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px'>"
    "Built with AWS S3 · Lambda · API Gateway · Streamlit | IST 615 Spring 2026"
    "</div>",
    unsafe_allow_html=True
)




