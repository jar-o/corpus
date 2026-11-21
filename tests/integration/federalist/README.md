# Case study: Federalist Papers

In Claude's own words, the [Federalist Papers](https://guides.loc.gov/federalist-papers/full-text) is a very well understood and deeply anlayzed text that is part of its training set. That means it is quite adept at querying and answering questions about the text.

For example, the test prompt (see below) is specifically phrased with words that cannot be directly searched in the Papers, e.g. the word "peaceful". Yet Claude knows to substitute *"keeping an army"* for a regex that will match *"standing army"* or its plural:

```bash
grep -n -i "standing arm\|time of peace\|in peace"
```

All that to say -- we are trying to give Claude a fair advantage here. Even so, using `corpus` to front-run with **free** local old school NLP gives you a **45.5%** cost savings for the same prompt:

| claude | claude+corpus | savings
|--------|---------------|--------
| $0.3089 | $0.1684 | 45.5%
|157,619 tokens | 52,302 tokens | 66.8%

## Test setup

Some convience functions used. This is designed as a manual test.

```bash
# run claude straight against the text files, no assistance
dbgstraight()
{
    claude -p "$1" --output-format stream-json --verbose --permission-mode auto | tee run.jsonl | jq -r 'select(.type=="assistant") | .message.content[] | select(.type=="tool_use") | "\(.name): \(.input.command // .input.file_path // .input.pattern // (.input|tostring))"'
}

# expects you to have first run `corpus .` manually in this folder
dbgcorpus()
{
    claude -p "use the command 'corpus --find' for the following prompt and answer from its output first if you can. the prompt: $1" --allowedTools "Bash(corpus:*)" --output-format stream-json --verbose | tee run.jsonl | jq -r 'select(.type=="assistant") | .message.content[] | select(.type=="tool_use") | "\(.name): \(.input.command // .input.file_path // .input.pattern // (.input|tostring))"'
}
```

## Claude / Opus 5.5 

This is the straight, unadorned AI run. Meaning Claude uses use whatever tools available to find and read the files, comprehend the question, and provide a response.

```bash
> dbgstraight "In the documents in this folder, tell me how do Hamilton and Madison differ on keeping an army during peaceful times?"
```

### Run cost

This run cost **$0.3089**. It used **157,619 tokens** across 6 turns, with no permission denials.

Opus 5.5 is priced at $4 per million input tokens and $20 per million output tokens. Cache reads are $0.20 per million, and 1-hour cache writes are 2× input, so $8 per million.

| Type | Tokens | Rate per million | Cost | Share of cost |
|---|---:|---:|---:|---:|
| Input (uncached) | 10 | $4.00 | $0.0000 | 0% |
| Cache write (1h) | 29,464 | $8.00 | $0.2357 | **76%** |
| Cache read | 125,744 | $0.20 | $0.0251 | 8% |
| Output (incl. 120 thinking) | 2,401 | $20.00 | $0.0480 | 16% |
| **Total** | **157,619** | | **$0.3089** | |

---

### Prompt Response

Hamilton and Madison reach the same conclusion: the Constitution should not ban a peacetime army, and Congress should decide. Where they differ is in tone, in what worries them most, and in how much they trust the militia. Hamilton argues from practical need. Madison treats a standing army as a dangerous evil that is sometimes necessary.

### Where they agree

- **No constitutional ban.** Hamilton says "the necessity of leaving the matter to the discretion and prudence of the legislature" (`federalist-24.txt:154`). Madison asks, "How could a readiness for war in time of peace be safely prohibited, unless we could prohibit, in like manner, the preparations… of every hostile nation?" (`federalist-41.txt:91`).
- **Written bans don't work.** Hamilton points to Pennsylvania, whose bill of rights opposed peacetime armies but which raised troops anyway, as proof of "how unequal parchment provisions are to a struggle with public necessity" (`federalist-25.txt:151-172`). Madison says such a ban "plants in the Constitution itself necessary usurpations of power" (`federalist-41.txt:97`).
- **Disunion is the real danger.** Both say separate states or confederacies would each build armies, as Europe did (Hamilton in No. 8; Madison in No. 41).
- **The two-year funding limit is the main safeguard.** Congress can't fund an army for more than two years at a time (Hamilton in No. 26; Madison in No. 41).

### Where they differ

### 1. Emphasis: need versus danger

- **Hamilton** concentrates on concrete needs in peacetime. He names garrisons on the Western frontier against Indian raids, Britain and Spain building up forces nearby, and dockyards and arsenals that need defending (`federalist-24.txt:136-175`). He openly accepts that a small professional force amounts to "a standing army in time of peace; a small one, indeed, but not the less real for being small."
- **Madison** treats a standing army as a hazard to be kept as small as possible: "A standing force, therefore, is a dangerous, at the same time that it may be a necessary, provision… On an extensive scale its consequences may be fatal" (`federalist-41.txt:115-120`). He cites Rome, whose liberties were "the final victim to her military triumphs." His aim is "diminishing both the necessity and the danger" of relying on one.

### 2. How much army America actually needs

- **Hamilton** expects the force to grow as threats grow, with garrisons increased "in some ratio to the force by which our Western settlements might be annoyed."
- **Madison** uses Britain as his model. An island, it was protected from its neighbors and never needed a large peacetime army, and America's distance from Europe gives it "the same happy security." His conclusion is that "America united, with a handful of troops, or without a single soldier, exhibits a more forbidding posture… than America disunited, with a hundred thousand veterans" (`federalist-41.txt:126-129`).

### 3. The militia

- **Hamilton** doubts the militia can carry the country's defense. He calls the idea that it would be "at all times equal to the national defense" a "doctrine [that] had like to have lost us our independence," and argues that "the steady operations of war against a regular and disciplined army can only be successfully conducted by a force of the same kind" (`federalist-25.txt:135-149`). He does call the militia a "valuable and powerful auxiliary" (No. 26). He also argues in No. 29 that federal control of a well-regulated militia would reduce the need for an army.
- **Madison** treats the armed citizenry as the final check on a federal army. Even if Congress built the largest army the country could support, about 25,000–30,000 men, it would face "a militia amounting to near half a million of citizens with arms in their hands," organized by the state governments (`federalist-46.txt:195-205`).

### 4. Where the threat of tyranny comes from

- **Hamilton** downplays the risk that Congress and the President would plot to use an army against the people. He argues that the people "could not be long deceived" (No. 26) and that elected officials should be trusted with the tools to protect liberty (No. 25).
- **Madison** takes the fear more seriously as a structural problem. His answers are the Union itself, which removes the "pretext" for a large army, the funding limit, and the states and their militias as a counterweight (Nos. 41 and 46).

### In short

Hamilton argues that America needs peacetime forces and that the legislature should decide their size. Madison agrees a ban would be unwise, but he sees a standing army as a threat to liberty that the Union, the funding limit and an armed citizenry should keep small and controlled.

## Claude / Opus 5.5 / Corpus

This run queries via `corpus --find` using roughly the same prompt as in the straight run above, rather than directly interrogating the text files in the folder. Note that we pass the same, natural language prompt to `corpus` as we would to Claude itself; no special search phrasing is required.

```bash
> dbgcorpus "how do Hamilton and Madison differ on keeping an army during peaceful times?"
```

### Run cost

This run cost **$0.1684**. It used **52,302 tokens** over 2 turns, with no permission denials. Claude answered from the `corpus --find` output alone.

Opus 5.5 is priced at $4 per million input tokens and $20 per million output tokens. Cache reads are $0.20 per million, and 1-hour cache writes are 2× input, so $8 per million.

| Type | Tokens | Rate per million | Cost | Share of cost |
|---|---:|---:|---:|---:|
| Input (uncached) | 4 | $4.00 | $0.0000 | 0% |
| Cache write (1h) | 16,749 | $8.00 | $0.1340 | **80%** |
| Cache read | 34,173 | $0.20 | $0.0068 | 4% |
| Output (incl. 177 thinking) | 1,376 | $20.00 | $0.0275 | 16% |
| **Total** | **52,302** | | **$0.1684** | |

---

### Prompt Response

I answered this from the `corpus --find` output alone. Hamilton and Madison mostly agree here: neither wants the Constitution to ban an army in peacetime. They differ in how they argue it and what they stress.

### Hamilton: peacetime armies are a practical necessity, so leave it to Congress

- **Some army is always needed.** Garrisons on the Western frontier have been needed since before the Revolution. The militia "would not long, if at all, submit to be dragged from their occupations… in times of profound peace." A paid permanent force "amounts to a standing army in time of peace; a small one, indeed, but not the less real for being small." He concludes the matter belongs to "the discretion and prudence of the legislature" (`federalist-24.txt` L136-155).
- **A ban couldn't work.** He asks what "keeping them up" would even mean: "Shall it be a week, a month, a year?" (`federalist-25.txt` L75-91). Pennsylvania's Bill of Rights says armies "ought not to be kept up in time of peace," yet the state raised troops anyway, and Massachusetts did the same during an insurrection. In his words, "All violent policy… defeats itself" (`federalist-25.txt` L149-162).
- **The fear is inherited and overdone.** He traces it to an "hereditary impression" from Britain. He says it was wrongly turned against "the representatives of the people" through "an injudicious excess" (`federalist-26.txt` L91-104). He also notes the state provisions say "OUGHT NOT," not "SHALL NOT," which he reads as "a conflict between jealousy and conviction" (`federalist-26.txt` L117-125).
- **Union keeps armies small.** If the states split apart, each would need large standing armies, which push government "toward monarchy" (`federalist-08.txt` L64-81). A secure union could keep a small army that citizens see as "a necessary evil" (`federalist-08.txt` L120-136).

### Madison: banning armies is impossible, and the people can still resist them

- **Defence depends on what other nations do.** "How could a readiness for war in time of peace be safely prohibited, unless we could prohibit… the preparations… of every hostile nation?" He warns that "It is in vain to oppose constitutional barriers to the impulse of self-preservation." A ban would just get broken, and each breach becomes a precedent for "multiplied repetitions." He uses European history from Charles VII of France onward (`federalist-41.txt` L91-110).
- **The danger can be contained.** He imagines the worst case: a federal army as big as the country could support. Even then he puts it at only 25,000–30,000 men, facing "a militia amounting to near half a million of citizens with arms in their hands," with the state governments on the people's side (`federalist-46.txt` L191-204).

### The difference

Hamilton argues from practical needs, like frontier garrisons and domestic unrest, and from legislative discretion. He treats the fear of standing armies as mostly an overreaction. Madison argues from international self-preservation and from the danger of a ban that would inevitably be broken. Unlike Hamilton, he takes the danger to liberty seriously and answers it with the balance of power: armed citizens and the states would outnumber any federal army.
