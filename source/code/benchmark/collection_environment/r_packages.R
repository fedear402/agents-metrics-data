#!/usr/bin/env Rscript
# Plan, install, and verify the R package set for the collection environment.
#
# Usage (from the repository root):
#   Rscript --vanilla code/benchmark/collection_environment/r_packages.R plan
#   Rscript --vanilla code/benchmark/collection_environment/r_packages.R install
#
# plan    Reads the target list in packages.json and the CRAN binary index.
#         Writes r_lock.json: every package to add (targets and new hard
#         dependencies) with its exact version, and every target that would need
#         a newer version of an installed package (a conflict). A conflicting
#         target is not installed.
# install Installs the packages in r_lock.json into the first library path
#         (the R user library that `Rscript --vanilla` uses). It never replaces
#         an installed package. If CRAN has a different version than the lock,
#         it builds the locked version from the CRAN source archive. A GitHub
#         package is built from the tarball of its locked commit.
#         It stops if any installed package changes version.

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[[1]] else "plan"
here <- "code/benchmark/collection_environment"
repo <- "https://cloud.r-project.org"
options(repos = c(CRAN = repo), timeout = 600)

spec <- jsonlite::fromJSON(file.path(here, "packages.json"), simplifyVector = FALSE)$r
lock_path <- file.path(here, "r_lock.json")
hard <- c("Depends", "Imports", "LinkingTo")
base_names <- rownames(installed.packages(priority = "base"))

installed_versions <- function() {
  ip <- installed.packages(noCache = TRUE)
  ip <- ip[!duplicated(ip[, "Package"]), , drop = FALSE]
  setNames(ip[, "Version"], ip[, "Package"])
}

# Parse "pkg (>= 1.2)" entries of one DESCRIPTION field.
requirements <- function(field) {
  if (is.na(field) || !nzchar(field)) return(list())
  parts <- trimws(strsplit(gsub("\n", " ", field), ",")[[1]])
  out <- list()
  for (p in parts[nzchar(parts)]) {
    name <- trimws(sub("\\(.*", "", p))
    if (name == "R") next
    op <- NA_character_; ver <- NA_character_
    if (grepl("\\(", p)) {
      inner <- trimws(sub(".*\\((.*)\\).*", "\\1", p))
      op <- sub("^([<>=]+).*", "\\1", inner)
      ver <- trimws(sub("^[<>=]+", "", inner))
    }
    out[[length(out) + 1]] <- list(name = name, op = op, version = ver)
  }
  out
}

satisfies <- function(have, op, need) {
  if (is.na(op)) return(TRUE)
  cmp <- utils::compareVersion(have, need)
  switch(op, ">=" = cmp >= 0, ">" = cmp > 0, "==" = cmp == 0, "<=" = cmp <= 0, "<" = cmp < 0, TRUE)
}

plan <- function() {
  ap <- available.packages(type = "binary")
  have <- installed_versions()
  targets <- vapply(spec$cran, function(x) x$package, "")
  lock <- list(); conflicts <- list(); unavailable <- character()
  for (t in targets) {
    if (!t %in% rownames(ap)) { unavailable <- c(unavailable, t); next }
    closure <- unique(c(t, unlist(tools::package_dependencies(t, db = ap, which = hard, recursive = TRUE))))
    closure <- setdiff(closure, base_names)
    problems <- list(); additions <- character()
    for (p in closure) {
      if (p %in% names(have)) next
      if (!p %in% rownames(ap)) { problems[[length(problems) + 1]] <- list(package = p, reason = "not on CRAN"); next }
      additions <- c(additions, p)
    }
    # Every requirement from a package to be added must hold for installed packages.
    for (p in additions) {
      for (field in hard) for (r in requirements(ap[p, field])) {
        if (r$name %in% names(have) && !satisfies(have[[r$name]], r$op, r$version)) {
          problems[[length(problems) + 1]] <- list(package = r$name, installed = unname(have[[r$name]]),
                                                   required = paste(r$op, r$version), by = p)
        }
      }
    }
    if (length(problems)) { conflicts[[t]] <- problems; next }
    for (p in setdiff(additions, names(lock))) lock[[p]] <- list(version = unname(ap[p, "Version"]), source = "CRAN",
                                                                 target = p %in% targets, needed_by = t)
  }
  for (g in spec$github) {
    have_now <- installed_versions()
    deps <- vapply(g$depends, function(x) x, "")
    missing_deps <- setdiff(deps, c(names(have_now), names(lock)))
    if (length(missing_deps)) {
      conflicts[[g$package]] <- list(list(package = paste(missing_deps, collapse = ", "), reason = "dependency not locked"))
      next
    }
    lock[[g$package]] <- list(version = g$version, source = "GitHub", repo = g$repo, ref = g$ref,
                              target = TRUE, needed_by = g$package)
  }
  out <- list(r_version = as.character(getRversion()), repository = repo,
              binary_index = contrib.url(repo, "binary"), planned_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
              packages = lock, conflicts = conflicts, unavailable = as.list(unavailable))
  jsonlite::write_json(out, lock_path, auto_unbox = TRUE, pretty = TRUE)
  cat(sprintf("lock: %d packages, %d conflicts, %d unavailable\n", length(lock), length(conflicts), length(unavailable)))
  if (length(conflicts)) print(names(conflicts))
}

install <- function() {
  lock <- jsonlite::fromJSON(lock_path, simplifyVector = FALSE)
  before <- installed_versions()
  ap <- available.packages(type = "binary")
  lib <- .libPaths()[[1]]
  todo <- Filter(function(n) !n %in% names(before), names(lock$packages))
  cran <- Filter(function(n) lock$packages[[n]]$source == "CRAN", todo)
  same <- Filter(function(n) n %in% rownames(ap) && ap[n, "Version"] == lock$packages[[n]]$version, cran)
  if (length(same)) install.packages(same, lib = lib, type = "binary", dependencies = FALSE)
  for (n in setdiff(cran, same)) {
    v <- lock$packages[[n]]$version
    install.packages(sprintf("%s/src/contrib/Archive/%s/%s_%s.tar.gz", repo, n, n, v),
                     lib = lib, repos = NULL, type = "source")
  }
  for (n in Filter(function(n) lock$packages[[n]]$source == "GitHub", todo)) {
    p <- lock$packages[[n]]
    tarball <- file.path(tempdir(), paste0(n, "_", p$ref, ".tar.gz"))
    utils::download.file(sprintf("https://github.com/%s/archive/%s.tar.gz", p$repo, p$ref), tarball, mode = "wb")
    install.packages(tarball, lib = lib, repos = NULL, type = "source")
  }
  after <- installed_versions()
  changed <- Filter(function(n) after[[n]] != before[[n]], intersect(names(before), names(after)))
  removed <- setdiff(names(before), names(after))
  wrong <- Filter(function(n) !n %in% names(after) || after[[n]] != lock$packages[[n]]$version, names(lock$packages))
  if (length(changed) || length(removed)) stop("Installed packages changed: ", paste(c(changed, removed), collapse = ", "))
  if (length(wrong)) stop("Locked packages missing or at another version: ", paste(wrong, collapse = ", "))
  cat(sprintf("installed %d packages; no installed package changed\n", length(todo)))
}

switch(mode, plan = plan(), install = install(), stop("Unknown mode: ", mode))
