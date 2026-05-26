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

- M01 → M07 milestone generation
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
        # FULL ROADMAP JSON
        # =====================================================

        st.divider()

        st.header("📚 Full Roadmap JSON")

        st.json(roadmap_data)

        # =====================================================
        # MILESTONE JSON
        # =====================================================

        milestones = roadmap_data.get("Milestones", [])

        st.divider()

        st.header("🚀 Milestone JSON Output")

        st.json(milestones)

        # =====================================================
        # VISUAL MILESTONES
        # =====================================================

        st.divider()

        st.header("🎯 Milestone Visualization")

        for milestone in milestones:

            module_data = milestone.get("modules", {})

            expander_title = (
                f"{module_data.get('module_id', '')}"
                f" • "
                f"{milestone.get('title', '')}"
            )

            with st.expander(expander_title):

                st.subheader(
                    milestone.get("role", "")
                )

                st.write(
                    milestone.get("description", "")
                )

                st.info(
                    milestone.get("quote", "")
                )

                # =====================================================
                # TWO COLUMNS
                # =====================================================

                colA, colB = st.columns(2)

                # =====================================================
                # LEFT COLUMN
                # =====================================================

                with colA:

                    st.write("### Skills")

                    for skill in milestone.get("skills", []):

                        st.success(skill)

                    st.write("### Gaps")

                    for gap in milestone.get("gaps", []):

                        st.error(gap)

                # =====================================================
                # RIGHT COLUMN
                # =====================================================

                with colB:

                    st.write("### Career Progression")

                    for item in milestone.get(
                        "career_progression",
                        []
                    ):

                        st.write(f"✅ {item}")

                    st.write("### New Opportunities")

                    for item in milestone.get(
                        "new_opportunities",
                        []
                    ):

                        st.write(f"🚀 {item}")

                # =====================================================
                # MARKET VALUE
                # =====================================================

                st.write("### Market Value")

                st.warning(
                    milestone.get("market_value", "")
                )

                # =====================================================
                # WEEKLY BREAKDOWN
                # =====================================================

                st.write("### Weekly Progression")

                weeks = module_data.get("weeks", [])

                for week in weeks:

                    status = week.get("status")

                    if status == "completed":

                        emoji = "✅"

                    elif status == "active":

                        emoji = "🔥"

                    else:

                        emoji = "🔒"

                    st.write(
                        f"{emoji} Week "
                        f"{week.get('week')} "
                        f"— "
                        f"{week.get('focus')}"
                    )

                    skills = week.get("skills", [])

                    if skills:

                        st.caption(
                            ", ".join(skills)
                        )

                    mastery = week.get(
                        "mastery_at_end"
                    )

                    if mastery is not None:

                        st.progress(float(mastery))

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