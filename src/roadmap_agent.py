from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.pinecone_utils import retrieve_context,retrieve_icp_type
import os
import json
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM Configuration (Dual-Engine Fallback)
# ============================================================
def get_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    # If no OpenAI key → directly use Gemini
    if not openai_key:
        print("[LLM] No OpenAI key found → Using Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0.3
        )

    try:
        primary_llm = ChatOpenAI(
            api_key=openai_key,
            model="gpt-4o-mini",
            temperature=0.3
        )

        # Test call to validate API key
        primary_llm.invoke("ping")

        print("[LLM] ✓ OpenAI is valid → Using OpenAI with Gemini fallback")

        backup_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0.3
        )

        return primary_llm.with_fallbacks([backup_llm])

    except Exception as e:
        print(f"[LLM] OpenAI failed: {e}")
        print("[LLM] Switching completely to Gemini")

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0.3
        )


llm = get_llm()

# ============================================================
# Prompt Templates
# ============================================================
roadmap_prompt = PromptTemplate(
    input_variables=["context","icp_type"],
    template="""
You are a senior AI career strategist, roadmap architect, and career-state simulation engine for Vidya V3.

You are generating a deeply personalized career roadmap for ONE specific user.

The user context below contains:
- Resume/background
- Onboarding interview answers
- Career goals
- Skill gaps
- Learning preferences
- Work history
- Conversation summary
- Current learning discussions

USER CONTEXT:
{context}

USER ICP TYPE:
{icp_type}

YOUR TASK

Generate:
1. A personalized learning roadmap
2. A 7-stage career milestone progression (M01 → M07)
3. A milestone-aligned weekly plan with mastery tracking

The roadmap must feel:
- psychologically believable
- emotionally specific
- professionally realistic
- personalized to THIS exact user

ICP DETECTION RULES

Infer the user's ICP TYPE from the context.

ICP-A = College Student
Signals: Student, Fresher, Internship seeking, Placement preparation, Campus hiring, Learning fundamentals, Tier 2/3 college
Tone: aspirational, placement-focused, confidence-building
Career evolution: intern-ready, screening-ready, placement-ready, offer-ready, job-ready

ICP-B = Working Professional / Service Engineer
Signals: Already employed, Service engineer, Support engineer, Working professional, Upskilling, Promotion goals, Career-switch goals
Tone: practical, professional, growth-focused, switch/promotion-oriented
Career evolution: reporting-ready, promotion-ready, stakeholder-ready, switch-ready, leadership-ready

MILESTONE DESIGN RULES

Milestones represent IDENTITY EVOLUTION, NOT course completion.
Milestones MUST:
- evolve progressively
- feel realistic
- reflect career maturity
- match the user's actual background

Each milestone must include:
- milestone_id: integer 1-7 (unique)
- estimated_days: integer (should equal weeks_in_milestone * 7)
- role: short role title
- title: milestone name
- description: 1-2 sentences
- quote: short, emotionally believable 1-sentence quote
- skills: 3-6 concise skill tags
- gaps: 2-4 real gaps
- career_progression: 2-4 outcomes the user can now claim
- new_opportunities: 2-4 realistic opportunities unlocked
- market_value: salary range string (example: "3-5 LPA" or "INR 10000-20000/month")
- modules: see milestone module breakdown rules

Milestones should feel personalized, not generic. Avoid repeating titles, roles, or quotes.

MILESTONE MODULE BREAKDOWN RULES

Each milestone must include exactly ONE "modules" object.
modules.week_range.start and modules.week_range.end must match the weeks list.
Weeks must be contiguous and non-overlapping across milestones.
Each week object must include:
- week: integer
- focus: short focus statement
- skills: list of skill tags
- status: completed | active | locked (only ONE active week overall)
- mastery_at_end: number between 0 and 1 for completed weeks, null otherwise

Set modules.mastery to a number between 0 and 1 that reflects progress across its weeks.
Milestone "modules" are separate from the top-level "Modules" list. Output both.

MODULE RULES

- Beginner → 6-8 modules
- Intermediate → 8-10 modules
- Advanced → 6-8 modules

Each module:
- must contain 4-8 concise theoretical topics
- NO projects, NO coding assignments, NO implementation tasks
- MUST remain compatible with MCQ generation

KEEP EXISTING MODULE STRUCTURE UNCHANGED.

LANGUAGE RULES: ENGLISH ONLY. NO Hindi, NO Hinglish, NO Tamil, NO mixed language.

OUTPUT RULES: RETURN VALID JSON ONLY. NO markdown, NO explanations, NO code fences, NO extra text. RETURN RAW JSON ONLY.

RETURN JSON IN THIS EXACT STRUCTURE:

{{
    "CourseTitle": "string",
    "CourseDescription": "string",
    "DifficultyLevel": "Beginner|Intermediate|Advanced",
    "Weeks": 8,
    "LearningStyle": "theory",
    "WeeklyHours": 5,
    "Milestones": [
        {{
            "milestone_id": 1,
            "estimated_days": 14,
            "role": "string",
            "title": "string",
            "description": "string",
            "quote": "string",
            "skills": ["string"],
            "gaps": ["string"],
            "career_progression": ["string"],
            "new_opportunities": ["string"],
            "market_value": "string",
            "modules": {{
                "module_id": "M01",
                "week_range": {{ "start": 1, "end": 2 }},
                "mastery": 0.45,
                "weeks": [
                    {{
                        "week": 1,
                        "focus": "string",
                        "skills": ["string"],
                        "status": "completed",
                        "mastery_at_end": 0.35
                    }}
                ]
            }}
        }},
        {{
            "milestone_id": 2,
            "estimated_days": 7,
            "role": "string",
            "title": "string",
            "description": "string",
            "quote": "string",
            "skills": ["string"],
            "gaps": ["string"],
            "career_progression": ["string"],
            "new_opportunities": ["string"],
            "market_value": "string",
            "modules": {{
                "module_id": "M02",
                "week_range": {{ "start": 3, "end": 3 }},
                "mastery": 0.45,
                "weeks": [
                    {{
                        "week": 3,
                        "focus": "string",
                        "skills": ["string"],
                        "status": "locked",
                        "mastery_at_end": null
                    }}
                ]
            }}
        }},
        {{
            "milestone_id": 3,
            "estimated_days": 7,
            "role": "string",
            "title": "string",
            "description": "string",
            "quote": "string",
            "skills": ["string"],
            "gaps": ["string"],
            "career_progression": ["string"],
            "new_opportunities": ["string"],
            "market_value": "string",
            "modules": {{
                "module_id": "M03",
                "week_range": {{ "start": 4, "end": 4 }},
                "mastery": 0.45,
                "weeks": [
                    {{
                        "week": 4,
                        "focus": "string",
                        "skills": ["string"],
                        "status": "locked",
                        "mastery_at_end": null
                    }}
                ]
            }}
        }},
        {{
            "milestone_id": 4,
            "estimated_days": 7,
            "role": "string",
            "title": "string",
            "description": "string",
            "quote": "string",
            "skills": ["string"],
            "gaps": ["string"],
            "career_progression": ["string"],
            "new_opportunities": ["string"],
            "market_value": "string",
            "modules": {{
                "module_id": "M04",
                "week_range": {{ "start": 5, "end": 5 }},
                "mastery": 0.45,
                "weeks": [
                    {{
                        "week": 5,
                        "focus": "string",
                        "skills": ["string"],
                        "status": "locked",
                        "mastery_at_end": null
                    }}
                ]
            }}
        }},
        {{
            "milestone_id": 5,
            "estimated_days": 7,
            "role": "string",
            "title": "string",
            "description": "string",
            "quote": "string",
            "skills": ["string"],
            "gaps": ["string"],
            "career_progression": ["string"],
            "new_opportunities": ["string"],
            "market_value": "string",
            "modules": {{
                "module_id": "M05",
                "week_range": {{ "start": 6, "end": 6 }},
                "mastery": 0.45,
                "weeks": [
                    {{
                        "week": 6,
                        "focus": "string",
                        "skills": ["string"],
                        "status": "locked",
                        "mastery_at_end": null
                    }}
                ]
            }}
        }},
        {{
            "milestone_id": 6,
            "estimated_days": 7,
            "role": "string",
            "title": "string",
            "description": "string",
            "quote": "string",
            "skills": ["string"],
            "gaps": ["string"],
            "career_progression": ["string"],
            "new_opportunities": ["string"],
            "market_value": "string",
            "modules": {{
                "module_id": "M06",
                "week_range": {{ "start": 7, "end": 7 }},
                "mastery": 0.45,
                "weeks": [
                    {{
                        "week": 7,
                        "focus": "string",
                        "skills": ["string"],
                        "status": "locked",
                        "mastery_at_end": null
                    }}
                ]
            }}
        }},
        {{
            "milestone_id": 7,
            "estimated_days": 7,
            "role": "string",
            "title": "string",
            "description": "string",
            "quote": "string",
            "skills": ["string"],
            "gaps": ["string"],
            "career_progression": ["string"],
            "new_opportunities": ["string"],
            "market_value": "string",
            "modules": {{
                "module_id": "M07",
                "week_range": {{ "start": 8, "end": 8 }},
                "mastery": 0.45,
                "weeks": [
                    {{
                        "week": 8,
                        "focus": "string",
                        "skills": ["string"],
                        "status": "locked",
                        "mastery_at_end": null
                    }}
                ]
            }}
        }}
    ],
    "Modules": [
        {{
            "Week": 1,
            "ModuleName": "string",
            "Description": "string",
            "Topics": ["string"]
        }}
    ]
}}
"""
)


mcq_prompt = PromptTemplate(
    input_variables=["module_name", "module_description", "topics"],
    template="""
You are an expert quiz creator. Generate 5 high-quality multiple-choice questions for this learning module.

Module: {module_name}
Description: {module_description}
Topics Covered: {topics}

**REQUIREMENTS:**
- Questions should test understanding, not just memorization
- Each question must have 4 options (A, B, C, D)
- Only ONE correct answer per question
- Include a brief explanation for the correct answer

**LANGUAGE RULE (CRITICAL):**
- The entire response MUST be in ENGLISH ONLY
- DO NOT use Tamil, Hindi, Hinglish, or any other language
- DO NOT translate based on user context
- ALWAYS output in English

**Return ONLY valid JSON array:**

[
  {{
    "question": "Clear, specific question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Brief explanation of why this is correct"
  }}
]

**DO NOT include any text outside the JSON array.**
**DO NOT use markdown code blocks.**
**Return raw JSON only.**
"""
)

roadmap_chain = roadmap_prompt | llm | StrOutputParser()
mcq_chain = mcq_prompt | llm | StrOutputParser()

# ============================================================
# Logic Functions
# ============================================================

def generate_module_mcqs(module: dict) -> list:
    module_name = module.get("ModuleName", "Unknown Module")
    module_description = module.get("Description", "")
    topics = module.get("Topics", [])
    topics_str = " | ".join(topics) if isinstance(topics[0], str) else " | ".join(
        [t.get("TopicName", "") for t in topics]
    ) if topics else ""

    print(f"[MCQ] Generating quiz for: {module_name}")

    # Maximum retry attempts
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            result = mcq_chain.invoke({
                "module_name": module_name,
                "module_description": module_description,
                "topics": topics_str
            })

            clean_result = result.strip()
            
            # --- JSON REPAIR LOGIC START (MCQ) ---
            # Using a trick to avoid breaking the code parser with markdown backticks
            markdown_marker = "`" * 3
            if markdown_marker in clean_result:
                clean_result = clean_result.replace(markdown_marker + "json", "").replace(markdown_marker, "").strip()
            
            start_idx = clean_result.find('[')
            end_idx = clean_result.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                clean_result = clean_result[start_idx:end_idx]
            # --- JSON REPAIR LOGIC END ---

            # Attempt to parse JSON
            mcqs = json.loads(clean_result)
            
            # Validate that we have a list and it has 5 questions
            if not isinstance(mcqs, list):
                print(f"[MCQ] ✗ Expected list but got {type(mcqs)} for {module_name}, retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                continue
                
            # Check if we got exactly 5 questions
            if len(mcqs) == 5:
                # Don't add ai_quiz_id or sequence_number here anymore
                print(f"[MCQ] ✓ Generated {len(mcqs)} questions for {module_name}")
                return mcqs
            else:
                print(f"[MCQ] ⚠ Got {len(mcqs)} questions instead of 5 for {module_name}, retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                
        except json.JSONDecodeError as e:
            print(f"[MCQ] ✗ JSON Parse Error for {module_name} (attempt {retry_count + 1}/{max_retries}): {e}")
            print(f"[MCQ] Attempting to repair malformed JSON...")
            
            # Repair strategy 1: Remove trailing commas before closing brackets
            repaired_json = re.sub(r',\s*([\]}])', r'\1', clean_result)
            
            # Repair strategy 2: Add missing quotes around property names
            repaired_json = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired_json)
            
            # Repair strategy 3: Replace single quotes with double quotes
            repaired_json = repaired_json.replace("'", '"')
            
            # Repair strategy 4: Remove any control characters
            repaired_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', repaired_json)
            
            try:
                mcqs = json.loads(repaired_json)
                if isinstance(mcqs, list) and len(mcqs) == 5:
                    # Don't add ai_quiz_id or sequence_number here anymore
                    print(f"[MCQ] ✓ Successfully repaired JSON for {module_name}")
                    return mcqs
                else:
                    print(f"[MCQ] ⚠ After repair, got {len(mcqs) if isinstance(mcqs, list) else 'invalid'} questions, retrying...")
                    retry_count += 1
            except json.JSONDecodeError as e2:
                print(f"[MCQ] ✗ Repair failed for {module_name} (attempt {retry_count + 1}/{max_retries}): {e2}")
                
                # Repair strategy 5: Try to extract valid JSON array using regex
                try:
                    # Find anything that looks like a JSON array with objects
                    array_pattern = r'\[\s*\{.*?\}\s*\]'
                    json_match = re.search(array_pattern, clean_result, re.DOTALL)
                    if json_match:
                        extracted_json = json_match.group(0)
                        mcqs = json.loads(extracted_json)
                        if isinstance(mcqs, list) and len(mcqs) == 5:
                            # Don't add ai_quiz_id or sequence_number here anymore
                            print(f"[MCQ] ✓ Extracted valid JSON array for {module_name}")
                            return mcqs
                except:
                    pass
                
                retry_count += 1

        except Exception as e:
            print(f"[MCQ] ✗ Error generating MCQs for {module_name} (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1

    # If we've exhausted all retries, return empty list (no fallbacks)
    print(f"[MCQ] ✗ All {max_retries} attempts failed for {module_name}. No questions generated.")
    return []


def transform_to_backend_format(roadmap: dict) -> dict:
    chapters = []
    total_topics = 0
    total_quizzes = 0

    for idx, module in enumerate(roadmap.get("Modules", [])):
        topics_raw = module.get("Topics", [])

        # First create topic objects
        topic_objects = []
        for t_idx, topic in enumerate(topics_raw):
            title = topic if isinstance(topic, str) else topic.get("TopicName", "Untitled")
            topic_objects.append({
                "ai_topic_id": f"ai_topic_{uuid.uuid4().hex[:12]}",
                "title": title,
                "content_type": "video",
                "sequence_number": t_idx + 1
            })

        total_topics += len(topic_objects)
        
        # Generate MCQs
        quiz_questions = generate_module_mcqs(module)
        
        # Add ai_quiz_id and associate with topics
        if quiz_questions and topic_objects:
            # Distribute quiz questions across topics 
            for q_idx, quiz in enumerate(quiz_questions):
                quiz["ai_quiz_id"] = f"ai_quiz_{uuid.uuid4().hex[:12]}"
                quiz["sequence_number"] = q_idx + 1
                # Associate with a topic (distribute evenly)
                topic_idx = q_idx % len(topic_objects)
                quiz["ai_topic_id"] = topic_objects[topic_idx]["ai_topic_id"]
            
            total_quizzes += len(quiz_questions)
        elif quiz_questions:
            # If no topics, still add quizzes but without ai_topic_id
            for q_idx, quiz in enumerate(quiz_questions):
                quiz["ai_quiz_id"] = f"ai_quiz_{uuid.uuid4().hex[:12]}"
                quiz["sequence_number"] = q_idx + 1
            total_quizzes += len(quiz_questions)

        chapters.append({
            "ai_chapter_id": f"ai_chapter_{uuid.uuid4().hex[:12]}",
            "title": module.get("ModuleName", "Untitled Chapter"),
            "sequence_number": idx + 1,
            "topics": topic_objects,
            "quiz_questions": quiz_questions
        })

    course = {
        "ai_course_id": f"ai_course_{uuid.uuid4().hex[:12]}",
        "title": roadmap.get("CourseTitle", "Personalized Learning Roadmap"),
        "description": roadmap.get("CourseDescription", ""),
        "difficulty_level": roadmap.get("DifficultyLevel", "intermediate").lower(),
        "chapters": chapters
    }

    return {
        "course": course,
        "metadata": {
            "total_courses": 1,
            "total_chapters": len(chapters),
            "total_topics": total_topics,
            "total_quiz_questions": total_quizzes
        }
    }


def run_pipeline(user_id: str, trigger_mcq: bool = True, ai_session_id: str = None, ai_roadmap_id: str = None) -> dict:
    print(f"\n[ROADMAP AGENT] Starting for user: {user_id}")
    print("=" * 60)

    session_was_provided = bool(ai_session_id)

    if not ai_session_id:
        ai_session_id = f"ai_sess_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        print(f"[ROADMAP AGENT] ⚠ No session ID provided - generated: {ai_session_id}")
    else:
        print(f"[ROADMAP AGENT] ✓ Using session ID: {ai_session_id}")

    if not ai_roadmap_id:
        ai_roadmap_id = f"ai_roadmap_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print(f"  - ai_session_id: {ai_session_id}")
    print(f"  - ai_roadmap_id: {ai_roadmap_id}")

    print(f"\n[ROADMAP AGENT] Retrieving user context...")
    icp_type = retrieve_icp_type(user_id)
    if not icp_type:
        print("[ICP] Onboarding missing or icp_type not set")
        return {
            "error": "Please complete onboarding first.",
            "user_id": user_id,
            "ai_session_id": ai_session_id
        }

    print(f"[ICP] User classified as: {icp_type}")

    context = retrieve_context(user_id)

    if not context:
        print("[ROADMAP AGENT] ✗ No context found!")
        return {
            "error": "No user data found. Please complete onboarding first.",
            "user_id": user_id,
            "ai_session_id": ai_session_id
        }

    print(f"[ROADMAP AGENT] ✓ Context retrieved: {len(context)} chars")

    print(f"\n[ROADMAP AGENT] Generating roadmap with Dual-Engine (OpenAI -> Gemini)...")

    try:
        result = roadmap_chain.invoke({"context": context, "icp_type": icp_type})

        clean_result = result.strip()
        
        # --- JSON REPAIR LOGIC START (ROADMAP) ---
        markdown_marker = "`" * 3
        if markdown_marker in clean_result:
            clean_result = clean_result.replace(markdown_marker + "json", "").replace(markdown_marker, "").strip()
            
        start_idx = clean_result.find('{')
        end_idx = clean_result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            clean_result = clean_result[start_idx:end_idx]

        # --- JSON REPAIR LOGIC END ---

        roadmap_data = json.loads(clean_result)

        milestones = roadmap_data.get("Milestones", [])

        if len(milestones) != 7:
            raise ValueError("Exactly 7 milestones required")

        for idx, milestone in enumerate(milestones):
            if not isinstance(milestone, dict):
                raise ValueError("Each milestone must be an object")

            milestone_id = milestone.get("milestone_id", idx + 1)
            try:
                milestone_id = int(milestone_id)
            except (TypeError, ValueError):
                milestone_id = idx + 1

            milestone["milestone_id"] = milestone_id

            modules = milestone.get("modules", {})
            if isinstance(modules, list):
                modules = modules[0] if modules else {}

            if not isinstance(modules, dict):
                raise ValueError(
                    f"Milestone modules must be an object in {milestone_id}"
                )

            weeks = modules.get("weeks", [])
            if weeks is None:
                weeks = []
            if not isinstance(weeks, list):
                raise ValueError(
                    f"Milestone weeks must be a list in {milestone_id}"
                )

            week_range = modules.get("week_range", {})
            if not isinstance(week_range, dict):
                week_range = {}

            if weeks:
                first_week = weeks[0].get("week")
                last_week = weeks[-1].get("week")
                if isinstance(first_week, int) and isinstance(last_week, int):
                    week_range.setdefault("start", first_week)
                    week_range.setdefault("end", last_week)

            if "start" in week_range and "end" in week_range:
                modules["week_range"] = week_range
                if milestone.get("estimated_days") in (None, ""):
                    try:
                        start_week = int(week_range["start"])
                        end_week = int(week_range["end"])
                        milestone["estimated_days"] = max(
                            0, (end_week - start_week + 1) * 7
                        )
                    except (TypeError, ValueError):
                        pass

            if modules.get("mastery") is None:
                mastery_values = [
                    week.get("mastery_at_end")
                    for week in weeks
                    if isinstance(week.get("mastery_at_end"), (int, float))
                ]
                if mastery_values:
                    modules["mastery"] = round(
                        sum(mastery_values) / len(mastery_values), 2
                    )

            modules["weeks"] = weeks
            milestone["modules"] = modules



        print(f"[ROADMAP AGENT] ✓ Generated: {roadmap_data.get('CourseTitle')}")
        print(f"[ROADMAP AGENT]    Modules: {len(roadmap_data.get('Modules', []))}")

        print(f"\n[ROADMAP AGENT] Transforming to backend format & generating MCQs...")
        roadmap_structure = transform_to_backend_format(roadmap_data)

        meta = roadmap_structure["metadata"]
        print(f"[ROADMAP AGENT] ✓ Complete!")
        print(f"  Chapters: {meta['total_chapters']}, Topics: {meta['total_topics']}, Quizzes: {meta['total_quiz_questions']}")

        now = datetime.utcnow().isoformat()

        return {
            "id": str(uuid.uuid4()),
            "user_id": int(user_id),
            "ai_session_id": ai_session_id,
            "ai_roadmap_id": ai_roadmap_id,
            "title": roadmap_data.get("CourseTitle", "Personalized Learning Path"),
            "description": roadmap_data.get("CourseDescription", ""),
            "estimated_duration_weeks": roadmap_data.get("Weeks", 6),
            "difficulty_level": roadmap_data.get("DifficultyLevel", "intermediate").lower(),
            "roadmap_structure": roadmap_structure,
            "milestones": roadmap_data.get("Milestones", []),
            "ai_metadata": {
                "generated_at": now,
                "weekly_hours": roadmap_data.get("WeeklyHours", 5),
                "learning_style": roadmap_data.get("LearningStyle", "theory"),
                "session_source": "pinecone" if session_was_provided else "generated",
                "generation_model": "roadmap-gen-v2.1",
                "personalization_score": 0.92
            },
            "status": "confirmed",
            "payment_id": None,
            "is_paid": False,
            "created_at": now,
            "updated_at": now,
            "confirmed_at": now,
            "published_at": None
        }

    except json.JSONDecodeError as e:
        print(f"[ROADMAP AGENT] ✗ JSON Parse Error: {e}")
        return {
            "error": f"Invalid JSON generated: {str(e)}",
            "user_id": user_id,
            "ai_session_id": ai_session_id,
            "raw_output": result[:500] if 'result' in locals() else ""
        }

    except Exception as e:
        print(f"[ROADMAP AGENT] ✗ Error: {e}")
        print(clean_result[:1000])
        import traceback
        traceback.print_exc()
        return {
            "error": f"Failed to generate roadmap: {str(e)}",
            "user_id": user_id,
            "ai_session_id": ai_session_id
        }
