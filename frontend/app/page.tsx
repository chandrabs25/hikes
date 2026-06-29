"use client";

import Link from "next/link";
import { FormEvent, useMemo, useState } from "react";

type Difficulty = "Easy" | "Easy-Moderate" | "Moderate" | "Moderate-Difficult" | "Difficult";

type Participant = {
  name: string;
  age: string;
  notes: string;
};

type Recommendation = {
  trek_id: string;
  title: string;
  recommendation: string;
  reasons: string[];
  tradeoffs: string[];
  person_specific_notes: string[];
};

type LlmRecommendation = {
  mode: "live";
  recommended: Recommendation[];
  comparison: {
    trek_id: string;
    title: string;
    best_fit_for: string[];
    concerns: string[];
  }[];
  questions_to_refine: string[];
  notes: string;
};

type CandidateCard = {
  trek_id: string;
  title: string;
  source_url: string | null;
  image_url: string | null;
  video_url: string | null;
};

type ShortlistResponse = {
  eligible_candidates: CandidateCard[];
  conditional_candidates: CandidateCard[];
  excluded: { reason: string; count: number }[];
  llm_recommendation: LlmRecommendation;
};

type ComparisonRow = {
  trek_id: string;
  title: string;
  image_url: string | null;
  video_url: string | null;
  difficulty: string | null;
  duration_days: number | null;
  distance_km: number | null;
  altitude_ft: number | null;
  age_range: { min: number | null; max: number | null };
  fitness: string | null;
  pickup: { city: string | null; time: string | null };
  dropoff: { city: string | null; time: string | null };
  offloading: boolean | null;
  cloakroom: boolean | null;
  accommodation: string | null;
};

type ComparisonResponse = {
  rows: ComparisonRow[];
};

type ChatCitation = {
  chunk_id: string;
  trek_id: string;
  trek_title: string;
  section_type: string;
  title: string;
  source_url: string | null;
  score: number | null;
};

type ChatResponse = {
  mode: "live";
  answer: string;
  suggested_followups: string[];
  citations: ChatCitation[];
  used_trek_ids: string[];
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: ChatCitation[];
  suggested_followups?: string[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const difficulties: Difficulty[] = ["Easy", "Easy-Moderate", "Moderate", "Moderate-Difficult", "Difficult"];
const durationOptions = [2, 4, 5, 6, 7, 8, 9, 10, 15];
const monthOptions = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December"
];

/* ─── Simple Markdown Renderer ──────────────────────────────── */
function normaliseChatText(raw: string) {
  let text = raw.trim();
  if (text.startsWith("```")) {
    text = text.replace(/^```(?:json)?\s*/i, "").replace(/```$/i, "").trim();
  }
  if (text.startsWith("{")) {
    try {
      const parsed = JSON.parse(text) as { answer?: unknown };
      if (typeof parsed.answer === "string") {
        text = parsed.answer;
      }
    } catch {
      // Leave non-JSON text alone.
    }
  }
  return text.replace(/\\n/g, "\n").trim();
}

function normaliseChatResponse(response: ChatResponse): ChatResponse {
  let answer = normaliseChatText(response.answer);
  let suggestedFollowups = response.suggested_followups;
  try {
    const parsed = JSON.parse(response.answer) as { answer?: unknown; suggested_followups?: unknown };
    if (typeof parsed.answer === "string") {
      answer = normaliseChatText(parsed.answer);
    }
    if (Array.isArray(parsed.suggested_followups) && suggestedFollowups.length === 0) {
      suggestedFollowups = parsed.suggested_followups.filter((item): item is string => typeof item === "string");
    }
  } catch {
    // Already handled as plain text.
  }
  return { ...response, answer, suggested_followups: suggestedFollowups };
}

function renderMarkdown(text: string) {
  const lines = normaliseChatText(text).split(/\n/);
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];
  let key = 0;

  function flushList() {
    if (listItems.length > 0) {
      elements.push(
        <ul key={key++} className="chatList">
          {listItems.map((item, i) => (
            <li key={i}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      listItems = [];
    }
  }

  function renderInline(line: string): React.ReactNode {
    // Split on **bold** markers
    const parts = line.split(/(\*\*[^*]+\*\*)/);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "") {
      flushList();
      continue;
    }
    if (/^([-•]|\d+[.)])\s/.test(trimmed)) {
      listItems.push(trimmed.replace(/^([-•]|\d+[.)])\s*/, ""));
    } else {
      flushList();
      elements.push(<p key={key++} className="chatPara">{renderInline(trimmed)}</p>);
    }
  }
  flushList();
  return elements;
}

function groupCitations(citations: ChatCitation[]) {
  const groups = new Map<
    string,
    { trek_title: string; source_url: string | null; sections: { key: string; label: string }[] }
  >();

  for (const citation of citations) {
    const group = groups.get(citation.trek_id) ?? {
      trek_title: citation.trek_title,
      source_url: citation.source_url,
      sections: []
    };
    const label = `${citation.section_type.replace(/_/g, " ")} · ${citation.title}`;
    const key = `${citation.section_type}::${citation.title}`;
    if (!group.sections.some((section) => section.key === key)) {
      group.sections.push({ key, label });
    }
    groups.set(citation.trek_id, group);
  }

  return Array.from(groups.entries())
    .slice(0, 3)
    .map(([trekId, group]) => ({
      trekId,
      ...group,
      sections: group.sections.slice(0, 4)
    }));
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTransientStatus(status: number) {
  return status === 408 || status === 425 || status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

async function fetchWithRetry(path: string, init?: RequestInit) {
  const delays = [0, 900, 1800, 3000];
  let lastError: unknown;
  for (let attempt = 0; attempt < delays.length; attempt += 1) {
    if (delays[attempt] > 0) {
      await sleep(delays[attempt]);
    }
    try {
      const response = await fetch(`${API_BASE}${path}`, {
        ...init,
        headers: {
          "Content-Type": "application/json",
          ...(init?.headers ?? {})
        }
      });
      if (!isTransientStatus(response.status) || attempt === delays.length - 1) {
        return response;
      }
      lastError = new Error(`Request failed with status ${response.status}`);
    } catch (error) {
      lastError = error;
      if (attempt === delays.length - 1) {
        throw error;
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Request failed");
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetchWithRetry(path, init);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail ?? `Request failed with status ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body as T;
}

/* ─── Inline SVG Icons ──────────────────────────────────────── */
function IconMountain() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 3l4 8 5-5 2 7H2L8 3z" />
    </svg>
  );
}

function IconUsers() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

function IconCompass() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
    </svg>
  );
}

function IconEdit() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

function IconSettings() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function IconArrowLeft() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="19" y1="12" x2="5" y2="12" />
      <polyline points="12 19 5 12 12 5" />
    </svg>
  );
}

function IconArrowRight() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

function IconSend() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function IconSparkle() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8L12 2z" />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconPlus() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function IconChat() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

/* ─── Step Config ───────────────────────────────────────────── */
const STEPS = [
  { label: "Group", icon: <IconUsers /> },
  { label: "Preferences", icon: <IconCompass /> },
  { label: "Trip goals", icon: <IconEdit /> },
  { label: "Details", icon: <IconSettings /> }
];

export default function Home() {
  const [screen, setScreen] = useState<"onboarding" | "results">("onboarding");
  const [participants, setParticipants] = useState<Participant[]>([
    { name: "", age: "", notes: "" }
  ]);
  const [travelMonth, setTravelMonth] = useState("December");
  const [targetDifficulty, setTargetDifficulty] = useState<Difficulty>("Easy-Moderate");
  const [durationDays, setDurationDays] = useState(6);
  const [textInput, setTextInput] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [altitudeCeiling, setAltitudeCeiling] = useState("");
  const [pickupCity, setPickupCity] = useState("");
  const [needsOffloading, setNeedsOffloading] = useState(false);
  const [themes, setThemes] = useState("");
  const [avoid, setAvoid] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [shortlist, setShortlist] = useState<ShortlistResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonRow[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  // New UI state for step wizard
  const [step, setStep] = useState(0);

  const canSubmit = useMemo(
    () => participants.every((person) => person.name.trim() && Number(person.age) > 0) && textInput.trim(),
    [participants, textInput]
  );
  const trekLinksById = useMemo(() => {
    const links = new Map<string, string>();
    if (!shortlist) {
      return links;
    }
    for (const card of [...shortlist.eligible_candidates, ...shortlist.conditional_candidates]) {
      if (card.source_url) {
        links.set(card.trek_id, card.source_url);
      }
    }
    return links;
  }, [shortlist]);
  const trekImagesById = useMemo(() => {
    const images = new Map<string, string>();
    if (!shortlist) {
      return images;
    }
    for (const card of [...shortlist.eligible_candidates, ...shortlist.conditional_candidates]) {
      if (card.image_url) {
        images.set(card.trek_id, card.image_url);
      }
    }
    return images;
  }, [shortlist]);
  const trekVideosById = useMemo(() => {
    const videos = new Map<string, string>();
    if (!shortlist) {
      return videos;
    }
    for (const card of [...shortlist.eligible_candidates, ...shortlist.conditional_candidates]) {
      if (card.video_url) {
        videos.set(card.trek_id, card.video_url);
      }
    }
    return videos;
  }, [shortlist]);

  function updateParticipant(index: number, key: keyof Participant, value: string) {
    setParticipants((current) =>
      current.map((person, personIndex) => (personIndex === index ? { ...person, [key]: value } : person))
    );
  }

  function addParticipant() {
    setParticipants((current) => [...current, { name: "", age: "", notes: "" }]);
  }

  function removeParticipant(index: number) {
    setParticipants((current) => current.filter((_, personIndex) => personIndex !== index));
  }

  function appendParticipantNote(index: number, note: string) {
    setParticipants((current) =>
      current.map((person, personIndex) => {
        if (personIndex !== index) {
          return person;
        }
        const currentNote = person.notes.trim();
        return {
          ...person,
          notes: currentNote ? `${currentNote}\n${note}` : note
        };
      })
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);
    setShortlist(null);
    setComparison([]);
    setSessionId("");
    setChatMessages([]);
    setChatInput("");
    setChatError("");

    try {
      const session = await apiRequest<{ session_id: string }>("/sessions", {
        method: "POST",
        body: JSON.stringify({ trip_name: "Trek recommendation" })
      });
      const payload = {
        participants: participants.map((person) => ({
          name: person.name.trim(),
          age: Number(person.age),
          ...(person.notes.trim() ? { notes: person.notes.trim() } : {})
        })),
        preferences: {
          travel_months: [travelMonth],
          target_difficulty: targetDifficulty,
          duration_days: durationDays,
          ...(altitudeCeiling ? { altitude_ceiling_ft: Number(altitudeCeiling) } : {}),
          ...(pickupCity.trim() ? { preferred_pickup_cities: [pickupCity.trim()] } : {}),
          ...(needsOffloading ? { needs_offloading: true } : {}),
          ...(themes.trim() ? { themes: themes.split(",").map((item) => item.trim()).filter(Boolean) } : {}),
          ...(avoid.trim() ? { avoid: avoid.split(",").map((item) => item.trim()).filter(Boolean) } : {})
        },
        text_input: textInput.trim()
      };
      await apiRequest(`/sessions/${session.session_id}/onboarding`, {
        method: "PUT",
        body: JSON.stringify(payload)
      });
      const nextShortlist = await apiRequest<ShortlistResponse>(`/sessions/${session.session_id}/shortlist`, {
        method: "POST"
      });
      const table = await apiRequest<ComparisonResponse>(`/sessions/${session.session_id}/comparison-table`).catch(() => ({
        rows: []
      }));
      setSessionId(session.session_id);
      setShortlist(nextShortlist);
      setComparison(table.rows);
      setScreen("results");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function askChat(question: string) {
    const trimmed = question.trim();
    if (!trimmed || !shortlist || !sessionId || chatLoading) {
      return;
    }
    setChatInput("");
    setChatError("");
    setChatLoading(true);
    setChatMessages((current) => [...current, { role: "user", content: trimmed }]);

    try {
      const recommendedIds = shortlist.llm_recommendation.recommended.map((trek) => trek.trek_id);
      const response = await apiRequest<ChatResponse>(`/sessions/${sessionId}/chat`, {
        method: "POST",
        body: JSON.stringify({
          question: trimmed,
          trek_ids: recommendedIds,
          max_chunks: 5
        })
      });
      const normalisedResponse = normaliseChatResponse(response);
      setChatMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: normalisedResponse.answer,
          citations: normalisedResponse.citations,
          suggested_followups: normalisedResponse.suggested_followups
        }
      ]);
    } catch (caught) {
      setChatError(caught instanceof Error ? caught.message : "Could not answer that question");
    } finally {
      setChatLoading(false);
    }
  }

  /* ─── Helpers for step validation ─────────────────────────── */
  const canProceedFromStep = (s: number) => {
    if (s === 0) return participants.every((p) => p.name.trim() && Number(p.age) > 0);
    if (s === 1) return true; // prefs always valid
    if (s === 2) return textInput.trim().length > 0;
    return true;
  };

  const totalSteps = showAdvanced ? 4 : 3;

  function handleNext() {
    if (step < totalSteps - 1) setStep(step + 1);
  }

  function handleBack() {
    if (step > 0) setStep(step - 1);
  }

  function getRankClass(index: number) {
    if (index === 0) return "rank1";
    if (index === 1) return "rank2";
    if (index === 2) return "rank3";
    return "rankOther";
  }

  /* ─── Prompt suggestions for text input ───────────────────── */
  const promptSuggestions = [
    "First Himalayan trek with kids — want something confidence-building",
    "We love snow and want stunning views without too much altitude",
    "Looking for a challenging trek with remote trail vibes",
    "Group has mixed fitness levels, need something everyone can enjoy",
    "Prefer quieter trails and fewer crowds",
    "Need something suitable for an older parent",
    "Comfortable with cold, but want to avoid technical terrain",
    "Want meadows, forests, and relaxed walking days"
  ];

  const personNoteHints = [
    "First Himalayan trek",
    "Gets tired on long climbs",
    "Loves snow",
    "Prefers easier days",
    "Has knee pain or past injury",
    "Gets anxious on exposed trails",
    "Comfortable with cold",
    "Needs a confidence-building trek"
  ];

  return (
    <main className="shell">
      {/* ─── Hero ─────────────────────────────────────────── */}
      <section className="hero">
        <div className="heroContent">
          <h1>Find treks that fit your whole group.</h1>
          <p className="heroSub">
            Tell us about your group and the trek experience you want — our AI matches you with the
            perfect Himalayan trek from Indiahikes.
          </p>
          <div className="heroActions">
            <Link className="btnSecondary" href="/trek-chat">
              <IconChat /> Ask about any trek
            </Link>
          </div>
        </div>
      </section>

      {/* ─── Onboarding ──────────────────────────────────── */}
      {screen === "onboarding" ? (
        loading ? (
          <div className="container">
            <div className="loadingOverlay">
              <div className="loadingDots">
                <span /><span /><span />
              </div>
              <p className="loadingText">Analyzing treks for your group…</p>
            </div>
          </div>
        ) : (
          <div className="container">
            <div className="wizardWrapper">
              {/* Step Bar */}
              <div className="stepsBar">
                {STEPS.slice(0, totalSteps).map((s, i) => (
                  <div className="stepItem" key={s.label}>
                    {i > 0 && (
                      <div className={`stepConnector${i <= step ? " completed" : ""}`} />
                    )}
                    <div
                      className={`stepItem${i === step ? " active" : ""}${i < step ? " completed" : ""}`}
                    >
                      <div
                        className={`stepCircle${i === step ? " active" : ""}${i < step ? " completed" : ""}`}
                      >
                        {i < step ? <IconCheck /> : i + 1}
                      </div>
                      <span className="stepLabel">{s.label}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Form wraps all steps for native validation on submit */}
              <form onSubmit={handleSubmit}>
                <div className="wizardPanel glass">
                  {/* Step 0: Group */}
                  {step === 0 && (
                    <div className="stepContent" key="step-0">
                      <h2 className="stepTitle">Who&rsquo;s trekking?</h2>
                      <p className="stepDesc">Add everyone in your group — names and ages help us personalize recommendations.</p>

                      <div className="peopleList">
                        {participants.map((person, index) => (
                          <div className="personCard" key={index}>
                            <div className="personCardFields">
                              <div className="personCardRow">
                                <div className="formGroup">
                                  <span className="formLabel">Name</span>
                                  <input
                                    className="formInput"
                                    value={person.name}
                                    onChange={(e) => updateParticipant(index, "name", e.target.value)}
                                    placeholder="Riya"
                                    required
                                  />
                                </div>
                                <div className="formGroup">
                                  <span className="formLabel">Age</span>
                                  <input
                                    className="formInput"
                                    type="number"
                                    min={1}
                                    value={person.age}
                                    onChange={(e) => updateParticipant(index, "age", e.target.value)}
                                    placeholder="10"
                                    required
                                  />
                                </div>
                              </div>
                              <div className="personCardNotes">
                                <div className="formGroup">
                                  <span className="formLabel">What should we know about this trekker?</span>
                                  <textarea
                                    className="formTextarea"
                                    value={person.notes}
                                    onChange={(e) => updateParticipant(index, "notes", e.target.value)}
                                    placeholder="First Himalayan trek. Gets tired on long climbs, but loves snow."
                                    rows={2}
                                  />
                                  <div className="personHints">
                                    <p className="hintLabel">Hints</p>
                                    {personNoteHints.map((hint) => (
                                      <button
                                        type="button"
                                        className="promptChip personHintChip"
                                        key={hint}
                                        onClick={() => appendParticipantNote(index, hint)}
                                      >
                                        {hint}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </div>
                            {participants.length > 1 && (
                              <div className="personActions">
                                <button
                                  className="removeBtn"
                                  type="button"
                                  onClick={() => removeParticipant(index)}
                                  aria-label="Remove person"
                                >
                                  ×
                                </button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>

                      <button className="addPersonBtn" type="button" onClick={addParticipant}>
                        <IconPlus /> Add another person
                      </button>
                    </div>
                  )}

                  {/* Step 1: Preferences */}
                  {step === 1 && (
                    <div className="stepContent" key="step-1">
                      <h2 className="stepTitle">Trek preferences</h2>
                      <p className="stepDesc">Set the basics — we&rsquo;ll use these to narrow down the best options.</p>

                      <div className="prefsGrid">
                        <div className="formGroup">
                          <span className="formLabel">Travel month</span>
                          <select
                            className="formSelect"
                            value={travelMonth}
                            onChange={(e) => setTravelMonth(e.target.value)}
                          >
                            {monthOptions.map((month) => (
                              <option key={month}>{month}</option>
                            ))}
                          </select>
                        </div>
                        <div className="formGroup">
                          <span className="formLabel">Target difficulty</span>
                          <select
                            className="formSelect"
                            value={targetDifficulty}
                            onChange={(e) => setTargetDifficulty(e.target.value as Difficulty)}
                          >
                            {difficulties.map((d) => (
                              <option key={d}>{d}</option>
                            ))}
                          </select>
                        </div>
                        <div className="formGroup">
                          <span className="formLabel">Trek duration</span>
                          <select
                            className="formSelect"
                            value={durationDays}
                            onChange={(e) => setDurationDays(Number(e.target.value))}
                          >
                            {durationOptions.map((days) => (
                              <option key={days} value={days}>
                                {days} days
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Step 2: Trip goals */}
                  {step === 2 && (
                    <div className="stepContent" key="step-2">
                      <h2 className="stepTitle">What kind of trek are you hoping for?</h2>
                      <p className="stepDesc">Describe the experience, comfort level, scenery, and concerns you want us to consider.</p>

                      <div className="formGroup">
                        <span className="formLabel">Trip goals and concerns</span>
                        <textarea
                          className="formTextarea"
                          value={textInput}
                          onChange={(e) => setTextInput(e.target.value)}
                          placeholder="We are taking our 10-year-old for a first Himalayan trek. We want snow and views, but it should feel confidence-building, not scary or exhausting."
                          rows={5}
                          required
                        />
                      </div>

                      <div className="promptChips">
                        <p className="hintLabel">Try mentioning</p>
                        {promptSuggestions.map((suggestion) => (
                          <button
                            type="button"
                            className="promptChip"
                            key={suggestion}
                            onClick={() =>
                              setTextInput((current) =>
                                current.trim() ? `${current.trim()}\n${suggestion}` : suggestion
                              )
                            }
                          >
                            {suggestion}
                          </button>
                        ))}
                      </div>

                      {!showAdvanced && (
                        <div style={{ marginTop: 20 }}>
                          <button
                            className="btnGhost"
                            type="button"
                            onClick={() => {
                              setShowAdvanced(true);
                            }}
                          >
                            <IconSettings /> Add optional details
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Step 3: Optional details (only if showAdvanced) */}
                  {step === 3 && showAdvanced && (
                    <div className="stepContent" key="step-3">
                      <h2 className="stepTitle">Optional details</h2>
                      <p className="stepDesc">These help us fine-tune, but aren&rsquo;t required.</p>

                      <div className="advancedGrid">
                        <div className="formGroup">
                          <span className="formLabel">Altitude ceiling (ft)</span>
                          <input
                            className="formInput"
                            type="number"
                            value={altitudeCeiling}
                            onChange={(e) => setAltitudeCeiling(e.target.value)}
                            placeholder="13000"
                          />
                        </div>
                        <div className="formGroup">
                          <span className="formLabel">Preferred pickup city</span>
                          <input
                            className="formInput"
                            value={pickupCity}
                            onChange={(e) => setPickupCity(e.target.value)}
                            placeholder="Dehradun"
                          />
                        </div>
                        <label className="checkboxLabel">
                          <input
                            type="checkbox"
                            checked={needsOffloading}
                            onChange={(e) => setNeedsOffloading(e.target.checked)}
                          />
                          <span>Need offloading</span>
                        </label>
                        <div className="formGroup">
                          <span className="formLabel">Themes</span>
                          <input
                            className="formInput"
                            value={themes}
                            onChange={(e) => setThemes(e.target.value)}
                            placeholder="snow, meadows"
                          />
                        </div>
                        <div className="formGroup">
                          <span className="formLabel">Avoid</span>
                          <input
                            className="formInput"
                            value={avoid}
                            onChange={(e) => setAvoid(e.target.value)}
                            placeholder="crowds, technical terrain"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Error display */}
                  {error && <p className="error">{error}</p>}

                  {/* Wizard Navigation */}
                  <div className="wizardNav">
                    {step > 0 ? (
                      <button className="btnSecondary" type="button" onClick={handleBack}>
                        <IconArrowLeft /> Back
                      </button>
                    ) : (
                      <div className="spacer" />
                    )}

                    {step < totalSteps - 1 ? (
                      <button
                        className="btnPrimary"
                        type="button"
                        onClick={handleNext}
                        disabled={!canProceedFromStep(step)}
                      >
                        Next <IconArrowRight />
                      </button>
                    ) : (
                      <button
                        className="btnPrimary"
                        type="submit"
                        disabled={!canSubmit || loading}
                      >
                        <IconSparkle /> Find suitable treks
                      </button>
                    )}
                  </div>
                </div>
              </form>
            </div>
          </div>
        )
      ) : (
        /* ─── Results ───────────────────────────────────────── */
        <div className="resultsContainer">
          {shortlist ? (
            <>
              {/* Header */}
              <div className="resultsHeader">
                <div className="resultsHeaderLeft">
                  <p className="eyebrow">AI Shortlist</p>
                  <h2 className="sectionTitle">Recommended Treks</h2>
                </div>
                <div className="resultsHeaderRight">
                  <button className="btnSecondary" type="button" onClick={() => { setScreen("onboarding"); setStep(0); }}>
                    <IconArrowLeft /> Edit answers
                  </button>
                  <span className="trekCount">
                    <IconMountain /> {shortlist.llm_recommendation.recommended.length} treks
                  </span>
                </div>
              </div>

              {/* Recommendation Cards */}
              <div className="recommendations">
                {shortlist.llm_recommendation.recommended.map((trek, index) => (
                  <article className="recCard glass" key={trek.trek_id}>
                    <div className="recCardHeader">
                      <div className={`rankBadge ${getRankClass(index)}`}>
                        {index + 1}
                      </div>
                      <div className="recCardInfo">
                        <h3>{trek.title}</h3>
                        <p>{trek.recommendation}</p>
                      </div>
                      {trekLinksById.get(trek.trek_id) ? (
                        <a
                          className="trekPageLink"
                          href={trekLinksById.get(trek.trek_id)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View trek page
                        </a>
                      ) : null}
                    </div>
                    <div className="recCardBody">
                      {(trekImagesById.get(trek.trek_id) || trekVideosById.get(trek.trek_id)) && (
                        <div className="trekDetailMedia">
                          {trekImagesById.get(trek.trek_id) ? (
                            <img
                              className="trekDetailImage"
                              src={trekImagesById.get(trek.trek_id)}
                              alt={`${trek.title} trek`}
                              loading="lazy"
                            />
                          ) : null}
                          {trekVideosById.get(trek.trek_id) ? (
                            <div className="trekDetailVideo">
                              <iframe
                                src={youtubeEmbedUrl(trekVideosById.get(trek.trek_id) ?? null)}
                                title={`${trek.title} video`}
                                loading="lazy"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                allowFullScreen
                              />
                            </div>
                          ) : null}
                        </div>
                      )}
                      <div className="recSection">
                        <h4>
                          <IconCheck /> Why it fits
                        </h4>
                        <ul>
                          {trek.reasons.map((reason) => (
                            <li key={reason}>{reason}</li>
                          ))}
                        </ul>
                      </div>
                      {trek.person_specific_notes.length > 0 && (
                        <div className="recSection personNotes">
                          <h4>
                            <IconUsers /> For your group
                          </h4>
                          <ul>
                            {trek.person_specific_notes.map((note) => (
                              <li key={note}>{note}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {trek.tradeoffs.length > 0 && (
                        <div className="recSection tradeoffs">
                          <h4>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                              <line x1="12" y1="9" x2="12" y2="13" />
                              <line x1="12" y1="17" x2="12.01" y2="17" />
                            </svg>
                            Tradeoffs
                          </h4>
                          <ul>
                            {trek.tradeoffs.map((tradeoff) => (
                              <li key={tradeoff}>{tradeoff}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </article>
                ))}
              </div>

              {/* Fit Summary */}
              {shortlist.llm_recommendation.comparison.length > 0 && (
                <section className="fitSection">
                  <p className="eyebrow">Fit analysis</p>
                  <h3 className="sectionTitle">How each trek fits</h3>
                  <div className="fitGrid">
                    {shortlist.llm_recommendation.comparison.map((item) => (
                      <article className="fitCard glass" key={item.trek_id}>
                        <h4>{item.title}</h4>
                        {item.best_fit_for.length > 0 && (
                          <>
                            <p className="fitLabel bestFit">Best fit for</p>
                            <ul>
                              {item.best_fit_for.map((fit) => (
                                <li key={fit}>{fit}</li>
                              ))}
                            </ul>
                          </>
                        )}
                        {item.concerns.length > 0 && (
                          <>
                            <p className="fitLabel concerns">Concerns</p>
                            <ul>
                              {item.concerns.map((concern) => (
                                <li key={concern}>{concern}</li>
                              ))}
                            </ul>
                          </>
                        )}
                      </article>
                    ))}
                  </div>
                </section>
              )}

              {/* Comparison Table */}
              {comparison.length > 0 && (
                <section className="tableSection">
                  <div className="tableHeaderBar">
                    <div>
                      <p className="eyebrow">Deterministic facts</p>
                      <h3 className="sectionTitle">Compare trek logistics</h3>
                    </div>
                    <span className="tableMeta">{comparison.length} recommended treks</span>
                  </div>
                  <div className="tableWrap">
                    <div className="tableScroll">
                      <table className="comparisonTable">
                        <thead>
                          <tr>
                            <th>Trek</th>
                            <th>Difficulty</th>
                            <th>Days</th>
                            <th>Distance</th>
                            <th>Altitude</th>
                            <th>Age</th>
                            <th>Fitness</th>
                            <th>Pickup</th>
                            <th>Dropoff</th>
                            <th>Offloading</th>
                            <th>Cloakroom</th>
                            <th>Stay</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparison.map((row) => (
                            <tr key={row.trek_id}>
                              <td>{row.title}</td>
                              <td>{row.difficulty ?? "Unknown"}</td>
                              <td>{row.duration_days ?? "Unknown"}</td>
                              <td>{row.distance_km ? `${row.distance_km} km` : "Unknown"}</td>
                              <td>{row.altitude_ft ? `${row.altitude_ft.toLocaleString()} ft` : "Unknown"}</td>
                              <td>{formatAgeRange(row.age_range)}</td>
                              <td>{row.fitness ?? "Unknown"}</td>
                              <td>{formatLogistics(row.pickup)}</td>
                              <td>{formatLogistics(row.dropoff)}</td>
                              <td>{row.offloading === null ? "Unknown" : row.offloading ? "Yes" : "No"}</td>
                              <td>{row.cloakroom === null ? "Unknown" : row.cloakroom ? "Yes" : "No"}</td>
                              <td>{row.accommodation ?? "Unknown"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </section>
              )}

              {/* Chat Panel */}
              <section className="chatSection">
                <div className="chatPanel glass">
                  <div className="chatPanelHeader">
                    <div>
                      <p className="eyebrow">Conversation</p>
                      <h3 className="sectionTitle">Chat with the trek assistant</h3>
                      <p className="chatIntro">
                        Ask follow-up questions about the suggested treks, compare options, or clarify fitness, season, safety, and itinerary details.
                      </p>
                    </div>
                    <span className="chatScope">{shortlist.llm_recommendation.recommended.length} treks in scope</span>
                  </div>

                  <div className="quickQuestions">
                    {[
                      "Which trek is easiest for a first timer?",
                      "Compare the snow experience.",
                      "What should we watch out for?",
                      "How do the itineraries differ?"
                    ].map((question) => (
                      <button
                        type="button"
                        className="quickBtn"
                        key={question}
                        onClick={() => askChat(question)}
                        disabled={chatLoading}
                      >
                        {question}
                      </button>
                    ))}
                  </div>

                  <div className="chatMessages">
                    {chatMessages.length > 0 ? (
                      chatMessages.map((message, index) => (
                        <article className={`chatBubble ${message.role}`} key={`${message.role}-${index}`}>
                          <div className="chatContent">{renderMarkdown(message.content)}</div>
                          {message.citations && message.citations.length > 0 && (
                            <div className="citationList citationGroups">
                              {groupCitations(message.citations).map((group) => (
                                <div className="citationGroup" key={group.trekId}>
                                  {group.source_url ? (
                                    <a className="citationGroupTitle" href={group.source_url} target="_blank" rel="noreferrer">
                                      {group.trek_title}
                                    </a>
                                  ) : (
                                    <span className="citationGroupTitle">{group.trek_title}</span>
                                  )}
                                  <div className="citationSections">
                                    {group.sections.map((section) => (
                                      <span className="citationChip" key={section.key}>
                                        {section.label}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                          {message.suggested_followups && message.suggested_followups.length > 0 && (
                            <div className="followups">
                              {message.suggested_followups.map((followup) => (
                                <button type="button" className="followupBtn" key={followup} onClick={() => askChat(followup)}>
                                  {followup}
                                </button>
                              ))}
                            </div>
                          )}
                        </article>
                      ))
                    ) : (
                      <div className="chatEmpty">
                        <IconChat />
                        <h4>Start a conversation about these recommendations</h4>
                        <p>Use the question box below to ask about itinerary, fitness, season, FAQs, safety, or tradeoffs between the suggested treks.</p>
                      </div>
                    )}
                    {chatLoading && (
                      <div className="typingIndicator">
                        <span /><span /><span />
                      </div>
                    )}
                  </div>

                  {chatError && <p className="error" style={{ margin: "0 24px" }}>{chatError}</p>}

                  <form
                    className="chatComposer"
                    onSubmit={(event) => {
                      event.preventDefault();
                      askChat(chatInput);
                    }}
                  >
                    <input
                      value={chatInput}
                      onChange={(event) => setChatInput(event.target.value)}
                      placeholder="Type a follow-up question about the suggested treks…"
                      aria-label="Ask a follow-up question about the suggested treks"
                      disabled={chatLoading}
                    />
                    <button className="sendBtn" type="submit" disabled={!chatInput.trim() || chatLoading}>
                      <IconSend />
                    </button>
                  </form>
                </div>
              </section>

              {/* Refine Questions */}
              {shortlist.llm_recommendation.questions_to_refine.length > 0 && (
                <section className="refineSection">
                  <h3>A few answers could improve this</h3>
                  <div className="refineButtons">
                    {shortlist.llm_recommendation.questions_to_refine.map((question) => (
                      <button type="button" className="refineBtn" key={question}>
                        {question}
                      </button>
                    ))}
                  </div>
                </section>
              )}

              {/* Notes */}
              {shortlist.llm_recommendation.notes && (
                <p className="notesBox">{shortlist.llm_recommendation.notes}</p>
              )}
            </>
          ) : (
            <div className="emptyState">
              <IconMountain />
              <h2>No shortlist yet</h2>
              <button className="btnPrimary" type="button" onClick={() => { setScreen("onboarding"); setStep(0); }}>
                Start onboarding
              </button>
            </div>
          )}
        </div>
      )}
    </main>
  );
}

function formatAgeRange(ageRange: ComparisonRow["age_range"]) {
  if (ageRange.min && ageRange.max) {
    return `${ageRange.min}-${ageRange.max}`;
  }
  if (ageRange.min) {
    return `${ageRange.min}+`;
  }
  if (ageRange.max) {
    return `Up to ${ageRange.max}`;
  }
  return "Unknown";
}

function formatLogistics(point: ComparisonRow["pickup"]) {
  if (point.city && point.time) {
    return `${point.city}, ${point.time}`;
  }
  return point.city ?? point.time ?? "Unknown";
}

function youtubeEmbedUrl(url: string | null) {
  if (!url) {
    return "";
  }
  const id =
    url.match(/[?&]v=([^&]+)/)?.[1] ??
    url.match(/youtu\.be\/([^?]+)/)?.[1] ??
    url.match(/embed\/([^?]+)/)?.[1];
  return id ? `https://www.youtube-nocookie.com/embed/${id}` : url;
}
