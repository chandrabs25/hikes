"use client";

import Link from "next/link";
import { FormEvent, useEffect, useMemo, useState } from "react";

type TrekListItem = {
  trek_id: string;
  title: string;
  source_url: string | null;
  image_url: string | null;
  video_url: string | null;
  difficulty: string | null;
  duration_days: number | null;
  distance_km: number | null;
  altitude_ft: number | null;
};

type TrekListResponse = {
  treks: TrekListItem[];
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
const maxSelectedTreks = 4;

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

  function renderInline(line: string): React.ReactNode {
    const parts = line.split(/(\*\*[^*]+\*\*)/);
    return parts.map((part, index) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={index}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  }

  function flushList() {
    if (listItems.length > 0) {
      elements.push(
        <ul key={key++} className="chatList">
          {listItems.map((item, index) => (
            <li key={index}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      listItems = [];
    }
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "") {
      flushList();
      continue;
    }
    if (/^([-•]|\d+[.)])\s/.test(trimmed)) {
      listItems.push(trimmed.replace(/^([-•]|\d+[.)])\s*/, ""));
      continue;
    }
    flushList();
    elements.push(
      <p key={key++} className="chatPara">
        {renderInline(trimmed)}
      </p>
    );
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

function IconArrowLeft() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="19" y1="12" x2="5" y2="12" />
      <polyline points="12 19 5 12 12 5" />
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

function IconSend() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function IconSearch() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function IconMountain() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 3l4 8 5-5 2 7H2L8 3z" />
    </svg>
  );
}

export default function TrekChatPage() {
  const [treks, setTreks] = useState<TrekListItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [userContext, setUserContext] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loadingTreks, setLoadingTreks] = useState(true);
  const [chatLoading, setChatLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoadingTreks(true);
    apiRequest<TrekListResponse>("/treks")
      .then((response) => {
        if (!cancelled) {
          setTreks(response.treks);
          setSelectedIds(response.treks.slice(0, 1).map((trek) => trek.trek_id));
        }
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Could not load treks");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingTreks(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedTreks = useMemo(
    () => selectedIds.map((id) => treks.find((trek) => trek.trek_id === id)).filter((trek): trek is TrekListItem => Boolean(trek)),
    [selectedIds, treks]
  );

  const filteredTreks = useMemo(() => {
    const term = query.trim().toLowerCase();
    const sortedTreks = [...treks].sort((a, b) => a.title.localeCompare(b.title));
    if (!term) {
      return sortedTreks;
    }
    return sortedTreks.filter((trek) => {
      const haystack = [
        trek.title,
        trek.difficulty ?? "",
        trek.duration_days ? `${trek.duration_days} days` : "",
        trek.altitude_ft ? `${trek.altitude_ft}` : ""
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(term);
    });
  }, [query, treks]);

  function toggleTrek(trekId: string) {
    setSelectedIds((current) => {
      if (current.includes(trekId)) {
        return current.filter((id) => id !== trekId);
      }
      if (current.length >= maxSelectedTreks) {
        return current;
      }
      return [...current, trekId];
    });
  }

  async function askChat(question: string) {
    const trimmed = question.trim();
    if (!trimmed || selectedIds.length === 0 || chatLoading) {
      return;
    }
    setChatInput("");
    setError("");
    setChatLoading(true);
    setMessages((current) => [...current, { role: "user", content: trimmed }]);

    try {
      const response = await apiRequest<ChatResponse>("/treks/chat", {
        method: "POST",
        body: JSON.stringify({
          trek_ids: selectedIds,
          question: trimmed,
          user_context: userContext.trim() || undefined,
          max_chunks: 5
        })
      });
      const normalised = normaliseChatResponse(response);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: normalised.answer,
          citations: normalised.citations,
          suggested_followups: normalised.suggested_followups
        }
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not answer that question");
    } finally {
      setChatLoading(false);
    }
  }

  function submitChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    askChat(chatInput);
  }

  return (
    <main className="shell trekChatShell">
      <section className="hero trekChatHero">
        <div className="heroContent">
          <div className="heroTopline">
            <Link className="btnSecondary" href="/">
              <IconArrowLeft /> Group trek finder
            </Link>
          </div>
          <h1>Ask about any Indiahikes trek.</h1>
          <p className="heroSub">
            Pick one trek for detailed answers, or select a few to compare itinerary, season, fitness, safety, and logistics.
          </p>
        </div>
      </section>

      <div className="trekChatContainer">
        <aside className="trekSelector glass">
          <div className="trekSelectorHeader">
            <p className="eyebrow">Trek library</p>
            <h2>Select treks</h2>
            <p>Choose up to {maxSelectedTreks} treks to keep answers focused.</p>
          </div>

          <label className="trekSearch">
            <IconSearch />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search by trek, difficulty, days..."
              aria-label="Search treks"
            />
          </label>

          <div className="selectedTreks">
            {selectedTreks.length > 0 ? (
              selectedTreks.map((trek) => (
                <button className="selectedTrekChip" type="button" key={trek.trek_id} onClick={() => toggleTrek(trek.trek_id)}>
                  {trek.title} ×
                </button>
              ))
            ) : (
              <span className="selectionEmpty">Select at least one trek</span>
            )}
          </div>

          <div className="trekList">
            {loadingTreks ? (
              <div className="loadingDots compactDots">
                <span /><span /><span />
              </div>
            ) : (
              filteredTreks.map((trek) => {
                const selected = selectedIds.includes(trek.trek_id);
                return (
                  <article className={`trekSelectCard${selected ? " selected" : ""}`} key={trek.trek_id}>
                    <button type="button" onClick={() => toggleTrek(trek.trek_id)} disabled={!selected && selectedIds.length >= maxSelectedTreks}>
                      <span className="trekSelectTitle">{trek.title}</span>
                      <span className="trekSelectMeta">
                        {trek.difficulty ?? "Difficulty unknown"} · {trek.duration_days ? `${trek.duration_days} days` : "Days unknown"}
                      </span>
                      <span className="trekSelectFacts">
                        {trek.distance_km ? `${trek.distance_km} km` : "Distance unknown"}
                        {trek.altitude_ft ? ` · ${trek.altitude_ft.toLocaleString()} ft` : ""}
                      </span>
                    </button>
                    {trek.source_url ? (
                      <a className="miniTrekLink" href={trek.source_url} target="_blank" rel="noreferrer">
                        View page
                      </a>
                    ) : null}
                  </article>
                );
              })
            )}
          </div>
        </aside>

        <section className="trekDirectChat">
          <div className="contextCard glass">
            <div>
              <p className="eyebrow">Optional context</p>
              <h2>Tell the assistant what to keep in mind</h2>
            </div>
            <textarea
              className="formTextarea"
              value={userContext}
              onChange={(event) => setUserContext(event.target.value)}
              placeholder="Example: We are beginners travelling in December with a 10-year-old. We want snow, but not a very exhausting trek."
              rows={3}
            />
          </div>

          <div className="chatPanel glass">
            <div className="chatPanelHeader">
              <div>
                <p className="eyebrow">Trek Q&A</p>
                <h3 className="sectionTitle">Chat about selected treks</h3>
                <p className="chatIntro">
                  The assistant searches only the selected trek content, then answers with citations from the matched sections.
                </p>
              </div>
              <span className="chatScope">{selectedTreks.length} selected</span>
            </div>

            <div className="quickQuestions">
              {[
                "What is the best season for these treks?",
                "Which one is easier for a first timer?",
                "Compare the itinerary and daily effort.",
                "What should we watch out for?"
              ].map((question) => (
                <button
                  type="button"
                  className="quickBtn"
                  key={question}
                  onClick={() => askChat(question)}
                  disabled={chatLoading || selectedIds.length === 0}
                >
                  {question}
                </button>
              ))}
            </div>

            <div className="chatMessages trekChatMessages">
              {messages.length > 0 ? (
                messages.map((message, index) => (
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
                  <IconMountain />
                  <h4>Select a trek and ask away</h4>
                  <p>Good questions include season, snow, difficulty, itinerary, fitness, safety, logistics, and comparisons.</p>
                </div>
              )}
              {chatLoading && (
                <div className="typingIndicator">
                  <span /><span /><span />
                </div>
              )}
            </div>

            {error && <p className="error trekChatError">{error}</p>}

            <form className="chatComposer" onSubmit={submitChat}>
              <input
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                placeholder="Ask about the selected trek or compare them..."
                aria-label="Ask about selected treks"
                disabled={chatLoading}
              />
              <button className="sendBtn" type="submit" disabled={!chatInput.trim() || selectedIds.length === 0 || chatLoading}>
                <IconSend />
              </button>
            </form>
          </div>
        </section>
      </div>
    </main>
  );
}
