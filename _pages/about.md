---
permalink: /
title: "Yejin Son (She/her)"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi! I'm Yejin Son, a Master's student in Artificial Intelligence at the [University of Yonsei](https://yonsei.ac.kr), [MIRLAB](https://mirlab.yonsei.ac.kr/) advised by [Youngjae Yu](https://yj-yu.github.io/home/). 

## Research Interests

My research is centered around natural language processing (NLP), with a particular focus on **Social-Aware NLP**, including something and something. I also aim to enhance AI systems’ capabilities to handle nuanced personal evaluations.

## Publications

<div style="display: flex; align-items: flex-start; margin-bottom: 24px;">
  <img src="/images/llm_safety_teaser.png" alt="Subtle Risks teaser" style="width: 140px; margin-right: 40px; border-radius: 5px;">
  <div>
    <p><strong>Subtle Risks, Critical Failures: A Framework for Diagnosing Physical Safety of LLMs for Embodied Decision Making</strong><br>
    <em>Yejin Son</em>, Minseo Kim, Sungwoong Kim, Seungju Han, Jian Kim, Dongju Jang, Youngjae Yu, Chan Young Park<br>
    <strong>Coming Soon</strong></p>
  </div>
</div>

<div style="display: flex; align-items: flex-start; margin-bottom: 24px;">
  <img src="/images/tom_teaser.png" alt="ToM teaser" style="width: 110px; margin-right: 20px; border-radius: 5px;">
  <div>
    <p><strong>Mind the Motions: Benchmarking Theory‑of‑Mind in Everyday Body Language</strong><br>
    Seungbeen Lee, Jinhong Jeong, Donghyun Kim, <em>Yejin Son</em>, Youngjae Yu<br>
    <strong>Coming Soon</strong></p>
  </div>
</div>

<div style="display: flex; align-items: flex-start; margin-bottom: 24px;">
  <img src="/images/book_norms_teaser.png" alt="Book Norms teaser" style="width: 110px; margin-right: 20px; border-radius: 5px;">
  <div>
    <p><strong>Reading Books is Great, But Not if You Are Driving! Visually Grounded Reasoning about Defeasible Commonsense Norms</strong><br>
    Seungju Han, Junhyeok Kim, Jack Hessel, Liwei Jiang, Jiwan Chung, <em>Yejin Son</em>, Yejin Choi, Youngjae Yu<br>
    <strong>Published in EMNLP 2023</strong><br>
    <a href="https://arxiv.org/abs/2310.10418">[Paper]</a></p>
  </div>
</div>

<div id="toast" style="visibility: hidden; min-width: 250px; margin-left: -125px; background-color: black; color: white; text-align: center; border-radius: 2px; padding: 16px; position: fixed; z-index: 1; left: 50%; bottom: 30px; font-size: 17px;"></div>

<script>
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(function() {
    showToast(text);
  }, function(err) {
    console.error('Could not copy text: ', err);
  });
}

function showToast(text) {
  var toast = document.getElementById("toast");
  toast.textContent = 'Citation copied to clipboard!';
  toast.style.visibility = "visible";
  setTimeout(function(){ toast.style.visibility = "hidden"; }, 1000);
}
</script>

------

Website last updated: May 22, 2025
