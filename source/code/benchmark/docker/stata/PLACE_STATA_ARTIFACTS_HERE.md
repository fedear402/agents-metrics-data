# Stata artifacts go here (not committed)

This directory is the drop point for enabling the Stata platform in the R3
agentic image. By default the Stata block in `../Dockerfile` is commented out
and nothing here is sent to the build context (see `../.dockerignore`).

To enable Stata, place BOTH of these here, then follow the "Stata dependency"
section of `../README.md`:

1. `Stata<NN>Linux64.tar.gz` — the x86-64 Linux installer tarball from StataCorp
   (download requires a valid license). Example: `Stata18Linux64.tar.gz`.
2. `stata.lic` — an activated **Linux** license file, OR have the serial / code /
   authorization ready to run `./stinit` during the build.

Plus, before building: written confirmation from StataCorp that your license
permits container / virtualized / ephemeral-host execution. Standard single-user
licenses are node-locked and do NOT cover this without StataCorp's approval. The
macOS license on the host does NOT cover a Linux container.

Do not commit the tarball or the license, and never push/share a Stata-enabled
image — it embeds a node-locked license.
