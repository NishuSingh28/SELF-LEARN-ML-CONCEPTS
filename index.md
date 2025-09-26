---
layout: default
title: "Home"
---

# Welcome to My Data Engineering Blog

Here you can find posts on data engineering, ML, and data science.

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a> - 
      <small>{{ post.date | date: "%B %d, %Y" }}</small>
    </li>
  {% endfor %}
</ul>
