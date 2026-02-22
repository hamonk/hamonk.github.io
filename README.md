# Your blog

This is the repository for your blog! Click *_config.yml* above to get started and fill in the details. Then click *index.md* and edit it to start creating your home page.

---

## Running locally (preview before pushing)

The site uses the `github-pages` gem, which mirrors the exact Jekyll version and plugins that GitHub Actions uses to build the site.

**First time setup**

```bash
gem install bundler   # if you don't have it yet
bundle install
```

**Start the dev server**

```bash
bundle exec jekyll serve --livereload
```

Then open [http://localhost:4000](http://localhost:4000).  
The `--livereload` flag automatically refreshes the browser whenever you save a file.

**Troubleshooting: Ruby 3.4+ `bigdecimal` error**

If you see `cannot load such file -- bigdecimal (LoadError)`, add this to your `Gemfile` and re-run `bundle install`:

```ruby
gem "bigdecimal"
```
