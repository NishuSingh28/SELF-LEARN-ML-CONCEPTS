---
layout: default
title: Time Series
permalink: /time-series/
---

# Time Series Posts

<ul>
  {% for post in site.categories.time-series %}
    <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
  {% else %}
    <li>No posts found in this category.</li>
  {% endfor %}
</ul>

<p><a href="{{ '/' | relative_url }}">← Back to Home</a></p>
<p><a href="{{ '/topics/' | relative_url }}">← Browse All Topics</a></p>
