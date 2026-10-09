"""Contract test: skill_manage create rejections must be SELF-REPAIRABLE.

Guards the 2026-10-09 monkeyhammer loop (session 02260176): the agent ratcheted
a description 132 -> 63 chars across 8 rejects because the error (a) never showed
the offending description and (b) the tool schema never stated the hard limit.

Contract: for an over-limit create, the returned error MUST include
1. the offending description verbatim (one look suffices to fix), and
2. the numeric limit (SKILL_PROMPT_DESC_LIMIT = 60).
"""

from tools.skill_manager_tool import SKILL_PROMPT_DESC_LIMIT, _validate_frontmatter


def _content(desc: str) -> str:
    return f"---\nname: x\ndescription: {desc}\n---\n\nbody\n"


def test_over_limit_error_is_self_repairable():
    desc = "A" * 94 + "."  # 95 chars, over the 60 budget
    err = _validate_frontmatter(_content(desc), new_skill=True)
    assert err, "validator must reject over-limit create"
    assert str(SKILL_PROMPT_DESC_LIMIT) in err, (
        f"error must state the limit ({SKILL_PROMPT_DESC_LIMIT}): {err[:120]}"
    )
    assert desc in err, (
        "error must echo the offending description verbatim so the agent can "
        "fix it in one pass instead of re-guessing"
    )


def test_under_limit_create_passes():
    err = _validate_frontmatter(
        _content("Search arXiv papers by keyword, author, or ID."), new_skill=True
    )
    assert err is None


def test_hard_1024_cap_still_enforced():
    err = _validate_frontmatter(_content("A" * 1100), new_skill=True)
    assert err and "1024" in err


def test_edit_path_skips_60_guard():
    # existing over-limit skills stay maintainable via patch/edit
    err = _validate_frontmatter(_content("A" * 300), new_skill=False)
    assert err is None
