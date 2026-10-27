# Local-only latexmk hook. Load from paper/manuscript/.latexmkrc.
# Export once after all successful targets, including an up-to-date build.
END {
    my $build_status = $?;
    if ($build_status == 0 && !$cleanup_mode && -f 'main.tex') {
        local $ENV{'UV_CACHE_DIR'} = $ENV{'UV_CACHE_DIR'} || 'build/uv-cache';
        my $export_status = system(
            'uv', 'run', '--no-sync', '--project', '../..',
            'python', '../../code/paper/export_sections.py'
        );
        if ($export_status != 0) {
            warn "Manuscript section export failed. See the exporter error above.\n";
            $build_status = 1;
        }
    }
    $? = $build_status;
}

1;
