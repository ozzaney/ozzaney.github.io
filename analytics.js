(function () {
  if (typeof gtag !== "function") return;

  const MEASUREMENT_ID = window.GA_MEASUREMENT_ID || "G-HQTH5E1KSJ";
  const pageName = document.body.dataset.page || "home";

  const PAGE_CONFIG = {
    home: {
      title: "Homepage · Yejin Son",
      label: "Home",
      path: "/",
      content_group: "Main Site",
    },
    cooking: {
      title: "Cooking Gallery · Yejin Son",
      label: "Cooking",
      path: "/cooking.html",
      content_group: "About Me",
    },
  };

  const pageConfig = PAGE_CONFIG[pageName] || {
    title: document.title,
    label: pageName,
    path: window.location.pathname,
    content_group: "Site",
  };

  gtag("config", MEASUREMENT_ID, {
    page_title: pageConfig.title,
    page_location: window.location.origin + pageConfig.path,
    page_path: pageConfig.path,
    content_group1: pageConfig.content_group,
    send_page_view: true,
  });

  function track(eventName, params) {
    gtag("event", eventName, { page_name: pageName, ...params });
  }

  function sectionForElement(element) {
    const section = element.closest("section[id], .intro-copy");
    if (!section) return "unknown";
    if (section.id === "home") return "Hero";
    if (section.classList.contains("intro-copy")) return "Introduction";
    if (section.id === "news") return "News";
    if (section.id === "research") return "Research";
    if (section.id === "about") return "About Me";
    if (section.classList.contains("collage")) return "Cooking Collage";
    return section.id || "unknown";
  }

  function linkCategory(link) {
    if (link.closest(".profile-links")) return "Profile";
    if (link.closest(".publication")) return "Publication";
    if (link.closest("#news")) return "News";
    if (link.closest(".hero")) return "Hero";
    if (link.closest(".intro-copy")) return "Introduction";
    if (link.closest("#about")) return "About Me";
    if (link.closest(".cooking-header, .cooking-footer")) return "Navigation";
    if (link.getAttribute("href") === "cooking.html") return "About Me";
    return "Content";
  }

  function linkLabel(link) {
    const aria = link.getAttribute("aria-label");
    if (aria) return aria;

    const publication = link.closest(".publication");
    if (publication) {
      const title = publication.querySelector("h3")?.textContent?.trim() || "Publication";
      const linkType = link.textContent.trim() || "link";
      return `${title} — ${linkType}`;
    }

    const text = link.textContent.replace(/\s+/g, " ").trim();
    if (text) return text.slice(0, 120);

    return link.hostname || link.getAttribute("href") || "link";
  }

  const SECTION_TARGETS = {
    home: [
      { selector: "#home", name: "Hero" },
      { selector: ".intro-copy", name: "Introduction" },
      { selector: "#news", name: "News" },
      { selector: "#research", name: "Research" },
      { selector: "#about", name: "About Me" },
    ],
    cooking: [{ selector: ".collage", name: "Cooking Collage" }],
  };

  const seenSections = new Set();
  const sectionTargets = SECTION_TARGETS[pageName] || [];

  if (sectionTargets.length && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting || entry.intersectionRatio < 0.35) return;

          const target = sectionTargets.find((item) => entry.target.matches(item.selector));
          if (!target || seenSections.has(target.name)) return;

          seenSections.add(target.name);

          const sectionSlug = target.name.toLowerCase().replace(/\s+/g, "-");
          track("section_view", { section_name: target.name });

          gtag("event", "page_view", {
            send_to: MEASUREMENT_ID,
            page_title: `${pageConfig.label} — ${target.name}`,
            page_location: `${window.location.origin}${pageConfig.path}#${sectionSlug}`,
            page_path: `${pageConfig.path}section/${sectionSlug}`,
            content_group1: pageConfig.content_group,
          });
        });
      },
      { threshold: [0.35, 0.6] }
    );

    sectionTargets.forEach((item) => {
      const node = document.querySelector(item.selector);
      if (node) observer.observe(node);
    });
  }

  document.addEventListener(
    "click",
    (event) => {
      const link = event.target.closest("a[href]");
      if (!link || link.classList.contains("pronounce-btn")) return;

      const href = link.getAttribute("href");
      if (!href || href.startsWith("#")) return;

      if (link.classList.contains("photo-swap-link")) return;

      track("link_click", {
        link_category: linkCategory(link),
        link_label: linkLabel(link),
        page_section: sectionForElement(link),
        link_url: link.href,
        outbound: link.hostname !== window.location.hostname ? "yes" : "no",
      });
    },
    true
  );

  document.querySelector(".pronounce-btn")?.addEventListener("click", () => {
    track("site_interaction", {
      interaction_name: "Pronunciation audio",
      page_section: "Hero",
    });
  });

  document.querySelectorAll(".photo-swap-link").forEach((link) => {
    link.addEventListener("click", () => {
      track("site_interaction", {
        interaction_name: "Hiking photo toggle",
        page_section: "About Me",
      });
    });
  });
})();
