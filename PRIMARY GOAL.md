You are an expert senior frontend engineer, UI/UX engineer, and motion designer.

Your task is to recreate the website referenced below as accurately as possible in a fully functional, production-quality website:

REFERENCE WEBSITE:
https://sienna-usage-407700.framer.app/

IMPORTANT: First, thoroughly analyze the entire reference website from top to bottom before writing the implementation.

## PRIMARY GOAL

Create an extremely high-fidelity recreation of the reference website's:

- Overall visual design
- Layout and section structure
- Spacing and alignment
- Color palette
- Typography hierarchy
- Font sizes and weights
- Borders and border radiuses
- Glass effects and transparency
- Background effects
- Gradients
- Shadows
- Cards and UI components
- Images and visual compositions
- Icons
- Buttons
- Navigation
- Hover states
- Scroll effects
- Entrance animations
- Continuous animations
- Interactive components
- Responsive behavior

The final result should visually feel as close to the reference website as possible while using properly licensed/original assets where necessary.

---

# STEP 1 — ANALYZE THE REFERENCE WEBSITE

Before coding, carefully inspect the entire website.

Analyze:

1. The navigation/header
2. The hero section
3. Every background element
4. Every decorative graphic
5. Every image placement
6. Every card and UI component
7. All platform/dashboard sections
8. Statistics sections
9. Feature sections
10. How-it-works sections
11. Security/compliance sections
12. Use-case cards
13. Integration sections
14. Testimonials
15. Pricing cards
16. Pricing toggle interactions
17. FAQ accordion
18. CTA sections
19. Footer
20. Mobile navigation and responsive layouts

Do not skip any section.

Create a complete structural understanding of the page before implementation.

---

# STEP 2 — VISUAL FIDELITY

Match the reference website as closely as possible.

Pay extremely close attention to:

### COLORS
- Identify the exact or closest possible background colors.
- Match primary and secondary colors.
- Match text colors.
- Match muted text colors.
- Match borders and transparent overlays.
- Match gradients precisely.

### TYPOGRAPHY
Analyze and reproduce:

- Font family
- Font weight
- Letter spacing
- Line height
- Heading scale
- Paragraph scale
- Button typography
- Navigation typography

Typography must feel visually identical in hierarchy and proportion to the reference.

### SPACING
Carefully reproduce:

- Container widths
- Section spacing
- Padding
- Margins
- Grid gaps
- Card spacing
- Alignment

Do not use arbitrary spacing values. Base spacing decisions on the visual proportions of the reference.

---

# STEP 3 — ANIMATIONS AND MOTION

Recreate ALL visible animations and interactions.

Carefully inspect the website for:

### PAGE LOAD ANIMATIONS
- Fade-ins
- Slide-ins
- Staggered text
- Image reveals
- Scale animations
- Blur-to-clear animations

### SCROLL ANIMATIONS
- Elements appearing while scrolling
- Parallax effects
- Section transitions
- Scroll-triggered fades
- Scroll-triggered movement
- Sticky sections if present

### CONTINUOUS ANIMATIONS
- Floating elements
- Rotating visuals
- Moving backgrounds
- Marquee/logo movement
- Animated decorative elements
- Dashboard animations

### HOVER INTERACTIONS
Recreate:

- Button hover effects
- Card hover effects
- Image hover effects
- Navigation hover effects
- Icon animations

Use smooth, premium animation curves.

Prefer:
- Framer Motion for React animations
- CSS animations where appropriate

All animations should feel smooth and performant.

Do not create random animations that do not exist in the reference.

---

# STEP 4 — RESPONSIVE DESIGN

The website must be fully responsive.

Implement optimized layouts for:

### Desktop
- 1440px+
- 1280px+

### Laptop
- Around 1024px–1280px

### Tablet
- Around 768px–1024px

### Mobile
- Around 320px–767px

On mobile:

- Adapt navigation appropriately.
- Create a functional mobile menu.
- Stack grids correctly.
- Resize typography intelligently.
- Preserve the visual hierarchy.
- Ensure cards do not overflow.
- Maintain appropriate spacing.

Do not simply shrink the desktop layout.

Create intentional responsive layouts.

---

# STEP 5 — COMPONENT ARCHITECTURE

Build the website using clean, reusable components.

Suggested structure:

/components
  Navbar
  Hero
  LogoMarquee
  PlatformOverview
  FeatureCard
  HowItWorks
  SecuritySection
  UseCases
  Integrations
  Stats
  Testimonials
  Pricing
  PricingCard
  FAQ
  CTA
  Footer

Keep the architecture modular and maintainable.

---

# STEP 6 — FUNCTIONALITY

Ensure all interactive elements actually work.

Implement:

### Navigation
- Smooth scrolling to sections
- Active/appropriate navigation behavior
- Mobile menu

### Buttons
- Functional links or placeholder actions
- Hover animations

### Pricing
- Monthly/yearly toggle
- Correct price changes if applicable

### FAQ
- Fully functional accordion
- Smooth expand/collapse animations

### Testimonials
- Recreate any carousel or animated behavior visible in the reference

### Marquees
- Smooth infinite looping
- No visible jump when looping

---

# STEP 7 — IMAGES AND ASSETS

Analyze every visual asset used in the reference.

For each asset:

- Determine its purpose
- Determine its placement
- Determine its dimensions and aspect ratio
- Reproduce the visual composition as closely as possible

If an asset cannot be legitimately reused, create or substitute an original asset that achieves the same visual role and aesthetic.

Do not leave empty placeholder boxes.

The final website should look complete and polished.

---

# TECHNICAL REQUIREMENTS

Build using:

- React
- Next.js preferred
- Tailwind CSS
- Framer Motion

Use:

- Clean semantic HTML
- Reusable components
- Responsive Tailwind utilities
- Optimized animations
- Proper image handling

Avoid:

- Excessive inline styles
- Messy monolithic components
- Hardcoded duplicated components
- Unnecessary dependencies

---

# IMPLEMENTATION PROCESS

Follow this workflow:

## PHASE 1
Analyze the reference website completely.

## PHASE 2
Create the complete page structure.

## PHASE 3
Implement the visual styling with high accuracy.

## PHASE 4
Implement all animations and interactions.

## PHASE 5
Implement responsive behavior.

## PHASE 6
Compare your result against the reference section by section.

For every section, check:

- Position
- Size
- Colors
- Typography
- Spacing
- Animation
- Responsiveness

Then refine discrepancies.

---

# IMPORTANT QUALITY STANDARD

Do NOT create a generic landing page inspired by the reference.

The objective is a high-fidelity recreation of the reference site's:

- Design language
- Layout
- Visual hierarchy
- Motion design
- Component structure
- User experience

Pay attention to small details.

Small details matter:

- Border opacity
- Subtle shadows
- Blur intensity
- Gradient positioning
- Letter spacing
- Button padding
- Card radius
- Animation duration
- Animation easing
- Section spacing

The website should feel premium, polished, and extremely close in visual quality and behavior to the reference.

---

# FINAL OUTPUT REQUIREMENTS

Provide a complete, runnable project.

Include:

1. All source code
2. Component files
3. Styling
4. Animation logic
5. Responsive behavior
6. Asset handling instructions
7. Setup instructions

Do not provide only an explanation or partial code.

Build the complete website implementation.

Before finishing, perform a final quality check and identify any visible differences between your implementation and the reference, then fix as many as possible.

PRIORITY ORDER:

1. Visual accuracy
2. Animation accuracy
3. Responsive accuracy
4. Functional interactions
5. Clean code architecture
6. Performance