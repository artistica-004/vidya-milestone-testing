import streamlit as st
import json
import sys
import os

# =====================================================
# FIX IMPORT PATH
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from src.roadmap_agent import roadmap_chain
from src.pinecone_utils import retrieve_context

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Vidya Milestone Testing Lab",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🚀 Vidya V3 — Milestone Testing Lab")

st.markdown("""
Internal testing dashboard for:

- Milestone generation
- ICP personalization testing
- learner-state testing
- onboarding-summary testing
- roadmap JSON validation
- weekly mastery progression
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🧠 User Persona Inputs")

icp_type = st.sidebar.selectbox(
    "ICP Type",
    [
        "high",
        "low"
    ]
)

learner_state = st.sidebar.selectbox(
    "Learner State",
    [
        "beginner",
        "intermediate",
        "advanced"
    ]
)

weekly_hours = st.sidebar.slider(
    "Weekly Hours",
    2,
    20,
    5
)

# =====================================================
# MODE SELECTION
# =====================================================

mode = st.radio(
    "Select Mode",
    [
        "Manual Testing",
        "Real User ID"
    ]
)

# =====================================================
# REAL USER MODE
# =====================================================

user_id = None
onboarding_summary = ""

if mode == "Real User ID":

    st.subheader("👤 Real User Testing")

    user_id = st.text_input(
        "Enter User ID",
        placeholder="Example: 123"
    )

# =====================================================
# MANUAL TESTING MODE
# =====================================================

else:

    st.subheader("📋 Learner Onboarding Summary")

    onboarding_summary = st.text_area(
        "Enter onboarding summary",
        height=250,
        placeholder="""
Example:

3rd year BTech IT student from tier 3 college.
Preparing for placements.
Interested in AI/ML.
Knows Python basics.
Weak in DSA and system design.
Wants internship in next 6 months.
"""
    )

# =====================================================
# GENERATE BUTTON
# =====================================================

generate_btn = st.button("🚀 Generate Personalized Roadmap")

# =====================================================
# GENERATION
# =====================================================

if generate_btn:

    try:

        # =====================================================
        # MANUAL TESTING MODE
        # =====================================================

        if mode == "Manual Testing":

            if not onboarding_summary.strip():

                st.error("Please enter onboarding summary")

                st.stop()

            context = f"""
USER PROFILE

ICP TYPE:
{icp_type}

LEARNER STATE:
{learner_state}

WEEKLY HOURS:
{weekly_hours}

ONBOARDING SUMMARY:
{onboarding_summary}
"""

        # =====================================================
        # REAL USER MODE
        # =====================================================

        else:

            if not user_id:

                st.error("Please enter user ID")

                st.stop()

            with st.spinner("Fetching user context from Pinecone..."):

                context = retrieve_context(user_id)

            if not context:

                st.error("No user context found in Pinecone")

                st.stop()

        # =====================================================
        # SHOW CONTEXT
        # =====================================================

        st.divider()

        st.header("🧠 Final Context Used")

        st.text_area(
            "Generated Context",
            value=context,
            height=250
        )

        # =====================================================
        # GENERATE ROADMAP
        # =====================================================

        with st.spinner("Generating roadmap + milestones..."):

            result = roadmap_chain.invoke({
                "context": context,
                "icp_type": icp_type
            })

        clean_result = result.strip()

        # =====================================================
        # CLEAN MARKDOWN
        # =====================================================

        if "```json" in clean_result:

            clean_result = (
                clean_result
                .replace("```json", "")
                .replace("```", "")
            )

        roadmap_data = json.loads(clean_result)

        st.success("✅ Roadmap Generated Successfully")

        # =====================================================
        # SUMMARY METRICS
        # =====================================================

        st.divider()

        st.header("📊 Roadmap Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Difficulty",
                roadmap_data.get("DifficultyLevel", "N/A")
            )

        with col2:

            st.metric(
                "Weeks",
                roadmap_data.get("Weeks", 0)
            )

        with col3:

            st.metric(
                "Milestones",
                len(roadmap_data.get("Milestones", []))
            )

        with col4:

            st.metric(
                "Modules",
                len(roadmap_data.get("Modules", []))
            )

        # =====================================================
        # COURSE OVERVIEW
        # =====================================================

        st.divider()

        st.header("📘 Course Overview")

        st.subheader(
            roadmap_data.get("CourseTitle", "")
        )

        st.write(
            roadmap_data.get("CourseDescription", "")
        )

        # =====================================================
        # VISUAL MILESTONES
        # =====================================================

        milestones = roadmap_data.get("Milestones", [])

        st.divider()

        st.header("🚀 Career Milestones")

        for milestone in milestones:

            module_data = milestone.get("modules", {})

            with st.container(border=True):

                top1, top2 = st.columns([4, 1])

                with top1:

                    st.subheader(
                        f'{module_data.get("module_id")} • {milestone.get("title")}'
                    )

                    st.caption(
                        milestone.get("role", "")
                    )

                with top2:

                    st.metric(
                        "Market Value",
                        milestone.get("market_value", "")
                    )

                st.write(
                    milestone.get("description", "")
                )

                st.info(
                    f'💬 "{milestone.get("quote", "")}"'
                )

                # =====================================================
                # SKILLS / GAPS / OPPORTUNITIES
                # =====================================================

                colA, colB, colC = st.columns(3)

                with colA:

                    st.markdown("### 🧠 Skills")

                    for skill in milestone.get("skills", []):

                        st.success(skill)

                with colB:

                    st.markdown("### ⚠ Gaps")

                    for gap in milestone.get("gaps", []):

                        st.error(gap)

                with colC:

                    st.markdown("### 🚀 Opportunities")

                    for opp in milestone.get(
                        "new_opportunities",
                        []
                    ):

                        st.info(opp)

                # =====================================================
                # CAREER PROGRESSION
                # =====================================================

                st.markdown("### 📈 Career Progression")

                for item in milestone.get(
                    "career_progression",
                    []
                ):

                    st.write(f"✅ {item}")

                # =====================================================
                # WEEKLY PROGRESSION
                # =====================================================

                st.markdown("### 📅 Weekly Progression")

                weeks = module_data.get("weeks", [])

                for week in weeks:

                    week_col1, week_col2 = st.columns([2, 5])

                    status = week.get("status")

                    if status == "completed":
                        emoji = "✅"

                    elif status == "active":
                        emoji = "🔥"

                    else:
                        emoji = "🔒"

                    with week_col1:

                        st.write(
                            f"{emoji} Week {week.get('week')}"
                        )

                    with week_col2:

                        st.write(
                            week.get("focus")
                        )

                        skills = week.get(
                            "skills",
                            []
                        )

                        if skills:

                            st.caption(
                                " • ".join(skills)
                            )

                        mastery = week.get(
                            "mastery_at_end"
                        )

                        if mastery is not None:

                            st.progress(float(mastery))

        # =====================================================
        # MODULE BREAKDOWN
        # =====================================================

        st.divider()

        st.header("📚 Learning Modules")

        modules = roadmap_data.get("Modules", [])

        for module in modules:

            with st.expander(
                f'Week {module.get("Week")} • {module.get("ModuleName")}'
            ):

                st.write(
                    module.get("Description")
                )

                st.markdown("### Topics")

                for topic in module.get(
                    "Topics",
                    []
                ):

                    st.write(f"• {topic}")

        # =====================================================
        # RAW JSON VIEW
        # =====================================================

        st.divider()

        with st.expander("🧾 Full Roadmap JSON"):

            st.json(roadmap_data)

        with st.expander("🧱 Milestones JSON"):

            st.json(milestones)

        # =====================================================
        # DOWNLOAD SECTION
        # =====================================================

        st.divider()

        st.header("⬇ Download Outputs")

        roadmap_json = json.dumps(
            roadmap_data,
            indent=2
        )

        milestone_json = json.dumps(
            milestones,
            indent=2
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                label="Download Full Roadmap JSON",
                data=roadmap_json,
                file_name="roadmap_output.json",
                mime="application/json"
            )

        with col2:

            st.download_button(
                label="Download Milestones JSON",
                data=milestone_json,
                file_name="milestones_output.json",
                mime="application/json"
            )

    except Exception as e:

        st.error(f"❌ Error: {str(e)}")