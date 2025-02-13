# Bamboost documentation

> The site bamboost.ch is built using [Fumadocs](fumadocs.vercel.app). The
> website source is [here](https://gitlab.com/zrlf/bamboost-docs). It is a static nextjs
> site.

## TODO

- [ ] document the next version of bamboost.
- [ ] change gitlab ci/cd to use the new workflow
- [ ] move hosting from the website repo to the main bamboost repo

## Info

- Documentation for `bamboost` is to be added here. The `./content` directory holds
  the markdown files building the documentation section of the site.

- Source code documentation is automatically generated using a custom extraction
  and rendering workflow, where we use `griffe` (from mkdocs) to extract the
  code, annotations, and docstrings from the source code, then dump it in a
  `json` file. The data is then rendered using custom react/nextjs components,
  specifically designed for the use with `fumadocs`. This is all done by code in
  the website repo.

## Build

To build the site, build the image from the Dockerfile, then run the container:

```bash
# build image
podman build -t bamboost-docs-builder .

# run container building the site
podman run --rm -v $(pwd)/..:/bamboost bamboost-docs-builder
```

The generated static site will be copied to `/public`, which is used by Gitlab
pages.

> [!note]
> The building of the site is done by the Gitlab CI/CD pipeline and hosted by
> Gitlab pages. It is triggered on tag creation.
