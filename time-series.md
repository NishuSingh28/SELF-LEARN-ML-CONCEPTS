---
layout: default
title: Time Series
permalink: /time-series/
---

# Time Series

<ul>
  {% for post in site.categories.time-series %}
    <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
