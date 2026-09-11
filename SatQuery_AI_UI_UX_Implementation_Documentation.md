# SatQuery AI — UI/UX Implementation Documentation

## Purpose

This document is the **single implementation specification** for rebuilding the SatQuery AI frontend.

The previous Bloom-inspired implementation produced an overly generic "AI slop" interface. **Do not recreate the Bloom two-panel layout, liquid glass cards, feature panels, community cards, or decorative glassmorphism.**

The revised direction is much simpler:

> **Keep the cinematic MotionSite background. Build a clean, premium AI interface inspired by the interaction patterns of ChatGPT and Claude, with a 21st.dev-inspired navigation bar, a Ruixen Moon Chat-style prompt composer, and pixel-font hero messaging.**

The website flow discussed previously must remain intact:

1. User lands on the cinematic SatQuery hero page.
2. The animated background establishes the atmosphere.
3. SatQuery AI is introduced in the center.
4. The hero messaging cycles between the required statements.
5. The user uploads satellite imagery and/or enters a natural-language query.
6. On submission, the application smoothly transitions into the analysis workspace.
7. The workspace behaves similarly to a ChatGPT-style conversation interface.
8. The cinematic MotionSite background remains visible behind the workspace, but becomes substantially darker/faded so the interface remains readable.

---

# 1. Core Design Philosophy

## The new direction

The interface should feel like:

- **ChatGPT / Claude** for the AI interaction model
- **21st.dev** for selected component aesthetics
- **Ruixen Moon Chat** for the hero prompt-composer atmosphere
- **Pixel Fonts / Geist Pixel** for the primary hero messaging
- **MotionSite cinematic background** for visual identity

It should **not** feel like:

- A generic SaaS dashboard
- A glassmorphism showcase
- A Bloom clone
- A card-heavy AI landing page
- A neon cyberpunk interface
- An overly decorated futuristic dashboard

### Design keywords

- Minimal
- Cinematic
- Intelligent
- Scientific
- Spatial
- Premium
- Dark
- Calm
- Precise
- Modern

---

# 2. Critical Rule: Do Not Use the Old Bloom UI

The previous specification included:

- Two-panel Bloom layout
- Liquid glass system
- Feature cards
- Community cards
- Social media pill
- Processing / Growth Archive cards
- Decorative quote section
- Strong glassmorphism everywhere

**Remove all of these concepts.**

The Bloom/MotionSite reference is now used primarily for its:

- Full-screen animated background
- Cinematic atmosphere
- Motion and visual pacing

The foreground UI must be redesigned from scratch.

---

# 3. Technology Requirements

The codebase should support:

- React
- TypeScript
- Tailwind CSS
- shadcn-compatible project structure
- Framer Motion for transitions and animation
- lucide-react for interface icons

Before implementation, inspect the existing repository.

## Required setup checks

Determine:

- Where reusable UI components currently live
- Where global styles are located
- Whether path aliases such as `@/components` are configured
- Whether Tailwind is already installed
- Whether TypeScript is already configured
- Whether shadcn conventions are already present

Preferred component structure:

```text
src/
├── components/
│   ├── ui/
│   │   ├── mini-navbar.tsx
│   │   ├── pixel-heading-word.tsx
│   │   └── ...
│   ├── HeroSection.tsx
│   ├── PromptComposer.tsx
│   ├── WorkspaceView.tsx
│   ├── Sidebar.tsx
│   └── ChatThread.tsx
├── assets/
├── App.tsx
└── index.css
```

If the project does not currently use `/components/ui`, create it and keep reusable, isolated UI primitives there.

---

# 4. Overall Application Architecture

Recommended structure:

```text
satquery_ui/
└── src/
    ├── components/
    │   ├── ui/
    │   │   ├── mini-navbar.tsx
    │   │   └── pixel-heading-word.tsx
    │   │
    │   ├── HeroSection.tsx
    │   ├── PromptComposer.tsx
    │   ├── UploadMenu.tsx
    │   ├── WorkspaceView.tsx
    │   ├── Sidebar.tsx
    │   ├── ChatThread.tsx
    │   ├── ChatMessage.tsx
    │   └── ExecutionSummary.tsx
    │
    ├── assets/
    │   └── (logos and imagery added later)
    │
    ├── App.tsx
    └── index.css
```

---

# 5. Application State and User Flow

The root application should manage two major views:

```ts
type AppView = "hero" | "workspace";
```

## Initial state

```text
Hero View
```

## After successful submission

```text
Hero View
     ↓
User enters query / uploads imagery
     ↓
Submit
     ↓
Smooth animated transition
     ↓
Workspace View
```

The transition must feel intentional and continuous.

Do not hard-cut to a completely different page.

The background should remain mounted throughout the transition.

---

# 6. Background System

## Primary background

Use the existing MotionSite-style cinematic animated/video background selected for SatQuery.

The background must:

- Fill the complete viewport
- Remain behind all UI
- Use `object-cover`
- Loop continuously
- Autoplay
- Be muted
- Support responsive resizing

Recommended structure:

```tsx
<div className="fixed inset-0 -z-10 overflow-hidden">
  <video
    autoPlay
    loop
    muted
    playsInline
    className="h-full w-full object-cover"
  >
    ...
  </video>
</div>
```

## Hero background state

On the hero page:

- Background should be visually prominent
- It should provide the main cinematic identity
- UI overlays should remain minimal

## Workspace background state

After entering the workspace:

- Keep the same background
- Darken it significantly
- Reduce its visual dominance
- Add an overlay for readability
- Do not remove the background completely

Suggested visual treatment:

```text
Hero:
Background opacity ≈ 100%

Workspace:
Background remains visible
+
Dark overlay
+
Reduced brightness
+
Optional subtle blur
```

The workspace must still feel connected to the landing page.

---

# 7. Navigation Bar — 21st.dev-Inspired Mini Navbar

Use the provided Mini Navbar component as the structural and interaction reference.

The navbar should be:

- Floating
- Horizontally centered
- Pill-shaped on desktop
- Dark
- Minimal
- Slightly translucent
- Subtly blurred
- Fixed near the top of the viewport
- Responsive

## Adapt the provided navbar for SatQuery

Replace the original generic content:

```text
Manifesto
Careers
Discover
LogIn
Signup
```

with SatQuery-specific navigation.

Recommended desktop navigation:

```text
SatQuery AI logo/name
Home
Features
How It Works
Why Us
GitHub Repo
```

The exact logo asset will be provided later.

### Important

Until the actual logo is uploaded:

- Use a simple placeholder mark
- Do not invent a permanent logo
- Keep the logo component easy to replace later

## Navbar interaction

Retain the elegant animated text behavior from the provided component.

On hover:

- Text should subtly animate vertically
- The alternate text layer becomes visible
- No exaggerated effects

## Mobile behavior

On mobile:

- Collapse navigation links into a menu
- Preserve the expanding navbar behavior
- Keep the header compact and usable
- Avoid covering the prompt composer

---

# 8. Hero Section

The hero should be extremely simple.

## Layout

Desktop:

```text
                 [ Floating Navbar ]


                    SatQuery AI


            Upload your satellite imagery
                 (Pixel Font)


          [ AI Prompt Composer Interface ]



          Cinematic MotionSite Background
```

The central experience is the AI interaction.

There should be **no side panels**.

There should be **no feature-card grid in the hero**.

There should be **no large decorative CTA buttons competing with the composer**.

---

# 9. SatQuery AI Branding

The hero should display:

```text
SatQuery AI
```

This should be centered prominently above the animated heading.

The final logo assets will be supplied later.

Until then:

- Use text branding
- Keep the structure prepared for a logo component
- Do not use unrelated stock logos

---

# 10. Pixel Font Hero Messaging

The main hero messaging must use the provided Pixel Heading component as the technical reference.

The required message is:

# Upload your satellite imagery

This should visually resemble the pixel typography shown in the 21st.dev reference.

## Font behavior

Use Geist Pixel font variables where supported.

The supplied PixelHeading component supports multiple font styles:

- Square
- Grid
- Circle
- Triangle
- Line

For SatQuery, start with a clean pixel style.

Recommended:

```tsx
<PixelHeading
  initialFont="square"
  hoverFont="circle"
  showLabel={false}
>
  Upload your satellite imagery
</PixelHeading>
```

However, the implementation must prioritize the final visual appearance over demonstrating every font variant.

## Important visual requirements

The hero heading should:

- Be centered
- Be white or near-white
- Use the pixel typography
- Have generous spacing around it
- Be large enough to establish identity
- Remain readable over the cinematic background

Do not add excessive text shadows.

Do not add rainbow or neon colors.

---

# 11. Hero Text Animation

The previously discussed flow included rotating hero messaging.

The hero messaging should support:

### State 1

```text
Ask the Earth. See the Answer.
```

### State 2

```text
Upload your satellite imagery
```

The second message is the primary final visual direction and should use the pixel-font treatment.

## Animation behavior

Use Framer Motion.

The transition should:

1. Display the first statement.
2. Hold long enough to be read.
3. Fade it out smoothly.
4. Optionally apply subtle blur during exit.
5. Bring in the second statement.
6. Avoid typewriter gimmicks unless they improve the final result.

The animation should feel calm and premium.

---

# 12. Hero Prompt Composer

This is the most important foreground component.

The prompt composer should combine the interaction philosophy of:

- ChatGPT
- Claude
- The Ruixen Moon Chat reference

It must **not** look like a generic form field.

## Overall shape

The composer should be:

- Large
- Horizontally centered
- Dark
- Rounded
- Minimal
- Spacious
- Clearly interactive

It should feel like the user is about to begin an AI conversation.

### Visual hierarchy

```text
[ Pixel Heading ]

[                                     ]
[     Ask anything / type request     ]
[                                     ]
[ upload / controls            send ↑ ]
```

---

# 13. Ruixen Moon Chat Influence

Use the Ruixen Moon Chat reference for the atmosphere around the prompt composer.

The reference demonstrates:

- A dark immersive canvas
- A large central AI composer
- A soft atmospheric visual element behind the composer
- Minimal pill controls beneath the composer

For SatQuery:

- Preserve the concept of atmospheric depth
- Adapt it to satellite/Earth imagery
- Do not copy unrelated purple branding
- Do not turn the UI into a neon-purple clone

The cinematic MotionSite background remains the main atmosphere.

If a soft supporting glow is used behind the composer, it must be:

- Subtle
- Neutral
- Derived from the background
- Not overly colorful

---

# 14. Prompt Composer Structure

The composer should contain:

## Left side

### Upload button

Use a plus or attachment icon.

Clicking it opens upload options.

Supported future upload types:

- Single satellite image
- SAR image
- Bi-temporal pair
- Optical + SAR pair

Do not make the upload workflow visually complicated on the first screen.

---

## Main input area

Use a textarea.

Suggested placeholder:

```text
Ask anything about your satellite imagery...
```

or:

```text
Describe what you want to analyze...
```

The final placeholder should feel conversational.

The input must support:

- Multi-line text
- Enter/submit behavior appropriate for AI chat
- Shift + Enter for new lines
- Auto-expanding height within reasonable limits

---

## Optional analysis control

A compact control can expose analysis modes or tools.

Do not clutter the main composer.

The interface should feel closer to ChatGPT/Claude:

```text
+   [ Ask anything...                         ]   [↑]
```

Optional tool controls can appear in a secondary row or compact menu.

---

## Right side

### Send button

Use a circular send button with an arrow icon.

States:

- Disabled when there is no input and no uploaded image
- Active when content exists
- Loading state after submission

The button should trigger the transition to the workspace.

---

# 15. Suggested Prompt Suggestions

Below the composer, optional compact suggestion chips may be displayed.

Examples:

```text
Detect changes
Analyze land use
Find flooded areas
Compare two images
```

These should be:

- Small
- Minimal
- Secondary to the composer
- Easy to remove if they clutter the design

Do not create a large dashboard of buttons.

---

# 16. Upload Flow

Clicking the upload control opens a small contextual menu.

Recommended options:

```text
Upload satellite image
Upload SAR image
Compare two images
Upload optical + SAR
```

Use:

- lucide-react icons
- A clean dark menu
- Subtle entrance animation
- Clear hover states

Do not use heavy glassmorphism.

---

# 17. Workspace After Submission

After the user submits a query, transition into the SatQuery AI workspace.

The workspace should be structurally inspired by ChatGPT.

## Layout

```text
┌─────────────────────────────────────────────────────────────┐
│ Sidebar │                                                  │
│         │                 Chat / Analysis                   │
│ New     │                                                  │
│ Analysis│     User query                                   │
│         │                                                  │
│ History │     SatQuery response                            │
│         │     ┌───────────────────────────────────────┐    │
│ Account │     │ Analysis / Results / Evidence         │    │
│         │     └───────────────────────────────────────┘    │
│         │                                                  │
│         │                Prompt Composer                    │
└─────────────────────────────────────────────────────────────┘
```

The cinematic background must remain visible behind the workspace at reduced prominence.

---

# 18. Workspace Sidebar

The left sidebar should be inspired by ChatGPT's layout philosophy.

## Top

- SatQuery AI branding
- New Analysis button

## Middle

Chat history grouped by time:

```text
Today
- Flood detection near Mumbai
- Compare Delhi imagery

Yesterday
- Forest cover analysis
- Urban expansion study
```

Use placeholder conversation names initially.

## Bottom

Account section.

Until real authentication is implemented:

- Use a placeholder avatar
- Use generic account information
- Keep it easy to connect to authentication later

---

# 19. Chat Thread

The central workspace should contain the conversation.

## User messages

User messages should be:

- Visually clean
- Clearly distinguishable
- More minimal than traditional oversized bubbles

## AI responses

SatQuery responses should support:

- Natural-language analysis
- Structured findings
- Satellite imagery
- Result images
- Confidence information
- Execution status

The AI response should feel like an intelligent remote-sensing assistant, not a generic chatbot.

---

# 20. Execution Summary

Each response may contain an expandable execution summary.

Example:

```text
Analysis completed

✓ Query classified
✓ Satellite input validated
✓ Appropriate model selected
✓ Analysis completed

Confidence: High
```

The detailed agent/model workflow should be:

- Collapsible
- Secondary
- Not forced into the user's main reading flow

Do not expose unnecessary technical complexity by default.

---

# 21. Image Evidence and Results

When the backend returns imagery or analysis results, the chat interface should support:

- Image previews
- Before/after comparison
- Captions
- Confidence labels
- Download/export actions where appropriate

Image results should be visually prominent when available.

Do not force every response into a text-only format.

---

# 22. Persistent Composer in Workspace

At the bottom of the workspace, retain a ChatGPT-style prompt composer.

The user should be able to:

- Ask follow-up questions
- Upload additional imagery
- Continue the same analysis conversation

The composer should visually connect the hero interaction with the workspace interaction.

---

# 23. Navigation Behavior

The navbar is primarily important on the landing/hero view.

Once inside the workspace:

- The sidebar becomes the primary navigation system
- The full marketing-style navbar may be hidden or simplified
- Avoid displaying two competing navigation systems

---

# 24. Responsive Requirements

## Desktop first

The initial design priority is desktop.

The hero should look excellent at:

- 1440px wide
- 1280px wide
- Large desktop displays

## Tablet

Adapt:

- Navbar spacing
- Heading size
- Composer width
- Workspace sidebar width

## Mobile

On mobile:

- Navbar collapses into the mobile menu
- Composer becomes nearly full width
- Hero typography scales down
- Sidebar becomes a drawer or collapsible panel
- Chat remains the primary interface

Do not simply shrink the desktop UI.

---

# 25. Animation Guidelines

Use Framer Motion for:

- Hero text transitions
- Hero-to-workspace transition
- Upload menu appearance
- Sidebar transitions
- Message entrance animations
- Navbar mobile expansion where needed

## Animation style

Animations should be:

- Smooth
- Fast enough to feel responsive
- Subtle
- Purposeful

Avoid:

- Excessive bouncing
- Constant floating cards
- Too many simultaneous animations
- Distracting parallax everywhere

---

# 26. Color System

The interface should primarily use:

```text
Near black
Dark charcoal
Soft gray
Off-white
White
```

The background imagery/video may naturally contain color.

The UI itself should remain restrained.

Avoid:

- Random gradients
- Neon purple
- Bright blue SaaS accents
- Rainbow effects
- Excessive colored badges

If semantic status colors are needed later:

- Keep them minimal
- Use them only for meaningful information

---

# 27. Typography

## Primary UI typography

Use a clean modern sans-serif suitable for:

- Navigation
- Inputs
- Chat
- Metadata
- Buttons

## Hero display typography

Use the Pixel Heading system for:

```text
Upload your satellite imagery
```

The provided PixelHeading implementation supports Geist Pixel font variants through CSS variables and Tailwind font utilities.

Ensure the required Geist font variables are configured before relying on classes such as:

```text
font-pixel-square
font-pixel-grid
font-pixel-circle
font-pixel-triangle
font-pixel-line
```

---

# 28. Pixel Heading Integration

Place the provided component at:

```text
/components/ui/pixel-heading-word.tsx
```

The component provides:

- Multiple pixel font variants
- Hover swapping
- Optional font cycling
- Keyboard accessibility
- Configurable heading element
- Configurable font states

For SatQuery's hero, do not overuse the interactive font cycling.

The preferred result is a polished, readable pixel heading rather than a typography demonstration.

---

# 29. Mini Navbar Integration

Place the provided navbar component at:

```text
/components/ui/mini-navbar.tsx
```

Adapt its content for SatQuery.

Preserve the useful interaction concepts:

- Centered floating placement
- Rounded desktop form
- Expanding mobile state
- Animated navigation text
- Subtle transparency
- Minimal dark styling

Do not preserve Bloom-specific labels.

---

# 30. Icons

Use `lucide-react`.

Potential icons include:

- Plus
- Paperclip
- ArrowUp
- Send
- Image
- Layers
- Scan
- History
- PanelLeft
- ChevronDown
- ChevronRight
- Download
- FileText
- Settings

Do not manually draw SVGs when Lucide provides an appropriate icon.

---

# 31. Assets

The user will provide:

- Final SatQuery logo
- Additional brand assets
- Any required imagery

Until then:

- Use placeholders sparingly
- Keep asset references modular
- Make logo replacement easy

Do not spend implementation effort creating a fake permanent identity.

---

# 32. Backend Integration

Preserve the existing backend API functionality.

The current interface should remain connected to the existing analysis endpoint beneath the redesigned UI.

The visual redesign must not break:

- Existing API calls
- File submission
- Inference requests
- Response handling

Recommended approach:

```text
UI Layer
   ↓
Prompt + Upload State
   ↓
Existing API / Analysis Request
   ↓
Loading State
   ↓
Chat-style Result Rendering
```

Do not replace working backend logic merely to redesign the frontend.

---

# 33. Loading States

After submission:

1. The message appears in the conversation.
2. The AI response enters a loading state.
3. Analysis progress is shown subtly.
4. Results replace or extend the loading response.

Possible loading text:

```text
Analyzing satellite imagery...
```

```text
Understanding your request...
```

```text
Running geospatial analysis...
```

Do not use overly theatrical loading animations.

---

# 34. Accessibility

Ensure:

- Buttons have accessible labels
- Keyboard navigation works
- Focus states are visible
- Contrast remains readable over the background
- Upload controls are clearly labeled
- Pixel typography does not reduce critical readability

---

# 35. Performance

Because the interface uses a full-screen video/animated background:

- Avoid unnecessary large animations
- Keep foreground rendering efficient
- Do not duplicate the background for each view
- Keep the video mounted across hero/workspace transitions
- Use optimized image rendering where appropriate

---

# 36. Explicit "Do Not Do" List

Claude must not:

- Recreate the Bloom two-panel layout
- Add liquid-glass cards everywhere
- Build generic SaaS feature sections into the hero
- Add decorative social-media widgets
- Use the Bloom flower imagery
- Use a purple neon Ruixen clone
- Create a dashboard before the user submits a query
- Make every element float independently
- Overuse gradients
- Add excessive cards
- Replace the cinematic background
- Break the existing backend integration

---

# 37. Implementation Priority

Implement in this order:

## Phase 1 — Foundation

1. Inspect current project structure
2. Verify TypeScript
3. Verify Tailwind
4. Verify shadcn-compatible component structure
5. Install required dependencies
6. Configure fonts

## Phase 2 — Background

1. Keep/create the full-screen MotionSite background
2. Ensure it persists across application states

## Phase 3 — Navigation

1. Integrate the adapted Mini Navbar
2. Implement desktop and mobile behavior

## Phase 4 — Hero

1. Build the centered SatQuery AI hero
2. Add the animated messaging
3. Integrate the pixel heading
4. Build the main prompt composer

## Phase 5 — Upload

1. Add upload interaction
2. Add supported upload options
3. Connect upload state to submission

## Phase 6 — Transition

1. Implement hero → workspace transition
2. Fade/darken the background

## Phase 7 — Workspace

1. Build sidebar
2. Build chat thread
3. Render user messages
4. Render AI analysis responses
5. Add persistent workspace composer

## Phase 8 — Backend

1. Preserve existing API calls
2. Connect real submission data
3. Render actual analysis results

## Phase 9 — Responsive polish

1. Desktop
2. Tablet
3. Mobile

---

# 38. Acceptance Criteria

The implementation is successful only if:

### Hero

- The MotionSite background remains the dominant visual environment
- SatQuery AI is centered
- The pixel-font heading says:
  **Upload your satellite imagery**
- The hero messaging flow works
- The composer feels similar in interaction quality to ChatGPT/Claude
- The composer incorporates the Ruixen Moon Chat visual inspiration without copying its purple branding
- The navbar follows the 21st.dev Mini Navbar aesthetic

### Transition

- Submission smoothly transitions into the workspace
- The background remains mounted and becomes darker/faded

### Workspace

- A ChatGPT-style sidebar exists
- Previous chats can be displayed
- The active conversation is central
- User and AI messages are clearly separated
- Analysis results can display imagery
- Follow-up prompts can be entered

### Technical

- Existing backend behavior remains intact
- TypeScript remains clean
- Components are modular
- Desktop is polished first
- Mobile behavior is functional

---

# Final Instruction to Claude

Do not interpret this document as a request to create a generic AI SaaS landing page.

The defining experience of SatQuery is:

> **A cinematic Earth/satellite environment where the user can naturally ask questions about the planet and upload satellite imagery for AI-powered analysis.**

The landing page should immediately lead into the AI interaction.

The composer is the product.

The workspace is the continuation of the same experience.

The background creates the world around it.

Keep the interface minimal enough that the user always knows what to do next:

# Upload imagery. Ask the Earth. See the answer.
