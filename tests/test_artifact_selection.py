"""Active-model discovery respects explicit selections and compatible bundles.

The tiny payloads below exercise artifact bookkeeping, never neural predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.artifacts as artifacts
from src.artifacts import ArtifactStore, atomic_json
from src.utils.filesystem import display_path


PRODUCT = 'aluminium_housing'
VIEW = 'inner_face'


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'artifacts', PRODUCT)


def publish(store, version='software-fixture-v1', *, view=VIEW, backend='efficientad'):
    with store.staging(view) as stage:
        (stage / 'payload.txt').write_text('Artifact-selection software fixture', encoding='utf-8')
        return store.publish(stage, {'version': version, 'view': view,
                                     'product': PRODUCT, 'backend': backend})


def pointer(store, backend, view=VIEW):
    return store.root / view / f'active-{backend}.json'


def rewrite_metadata(directory, update):
    path = directory / 'manifest.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata.update(update)
    atomic_json(path, metadata)


@pytest.mark.parametrize('requested,sibling', [('hybrid', 'efficientad'), ('efficientad', 'hybrid')])
def test_missing_neural_alias_uses_the_other_verified_selection_without_writes(store, requested, sibling):
    directory = publish(store)
    pointer(store, requested).unlink()
    selection = pointer(store, sibling).read_bytes()

    assert store.active(VIEW, requested) == directory

    assert pointer(store, sibling).read_bytes() == selection
    assert not pointer(store, requested).exists()


def test_explicit_different_versions_for_neural_modes_are_preserved(store):
    first = publish(store, 'selected-hybrid-version')
    second = publish(store, 'selected-efficientad-version')
    store.select(VIEW, first.name, 'hybrid')
    selections = {mode: pointer(store, mode).read_bytes() for mode in ('hybrid', 'efficientad')}

    assert store.active(VIEW, 'hybrid') == first
    assert store.active(VIEW, 'efficientad') == second
    assert all(pointer(store, mode).read_bytes() == data for mode, data in selections.items())


@pytest.mark.parametrize('contents', [
    b'{broken JSON',
    b'[]',
    b'{}',
    b'{"version":42}',
    b'{"version":"../other-product/model"}',
    b'{"version":"not-present"}',
])
def test_invalid_requested_pointer_does_not_fall_back_to_a_valid_sibling(store, contents):
    publish(store)
    pointer(store, 'hybrid').write_bytes(contents)

    with pytest.raises((ValueError, OSError)):
        store.active(VIEW, 'hybrid')

    assert pointer(store, 'hybrid').read_bytes() == contents


def test_corrupt_requested_bundle_does_not_fall_back_to_another_version(store):
    selected = publish(store, 'selected-version')
    publish(store, 'other-neural-version')
    store.select(VIEW, selected.name, 'hybrid')
    (selected / 'payload.txt').write_text('Changed after publication', encoding='utf-8')

    with pytest.raises(ValueError, match='integrity'):
        store.active(VIEW, 'hybrid')


def test_a_directory_in_place_of_the_requested_pointer_is_not_treated_as_missing(store):
    publish(store)
    requested = pointer(store, 'hybrid')
    requested.unlink()
    requested.mkdir()

    with pytest.raises(OSError):
        store.active(VIEW, 'hybrid')


def test_broken_requested_pointer_link_does_not_fall_back_to_a_valid_sibling(store):
    publish(store)
    requested = pointer(store, 'hybrid')
    requested.unlink()
    try:
        requested.symlink_to(requested.parent / 'missing-selection.json')
    except OSError:
        pytest.skip('Host does not permit creating symbolic links')

    with pytest.raises(FileNotFoundError):
        store.active(VIEW, 'hybrid')
    assert requested.is_symlink()


@pytest.mark.parametrize('verify_files', [True, False])
def test_bundle_directory_link_cannot_escape_its_product_and_view(store, tmp_path, verify_files):
    directory = publish(store)
    outside = tmp_path / 'outside-view'
    directory.rename(outside)
    try:
        directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Host does not permit creating symbolic links')

    with pytest.raises(ValueError, match='outside its product/view'):
        store.active(VIEW, 'hybrid', verify_files=verify_files)


@pytest.mark.parametrize('verify_files', [True, False])
@pytest.mark.parametrize('update,reason', [
    ({'product': 'different_product'}, 'product'),
    ({'view': 'ribbed_face'}, 'view'),
    ({'backend': 'geometry'}, 'EfficientAD weights'),
    ({'version': 'different-version'}, 'version'),
])
def test_alias_rejects_mismatched_bundle_context_with_or_without_digest_check(store, update, reason, verify_files):
    directory = publish(store)
    pointer(store, 'hybrid').unlink()
    rewrite_metadata(directory, update)

    with pytest.raises(ValueError, match=reason):
        store.active(VIEW, 'hybrid', verify_files=verify_files)


@pytest.mark.parametrize('backend', ['hybrid', 'efficientad'])
def test_geometry_is_not_silently_used_for_neural_inspection(store, backend):
    publish(store, backend='geometry')

    with pytest.raises(FileNotFoundError, match='No trained') as error:
        store.active(VIEW, backend)

    assert display_path(pointer(store, backend)) in str(error.value)
    assert 'Geometry mode' in str(error.value)


def test_saved_but_unselected_versions_do_not_auto_select_the_newest(store):
    publish(store, 'saved-version')
    pointer(store, 'hybrid').unlink()
    pointer(store, 'efficientad').unlink()

    with pytest.raises(FileNotFoundError, match='Model Settings'):
        store.active(VIEW, 'hybrid')


def test_neural_alias_resolution_stays_within_the_selected_face(store):
    face_a = publish(store)
    pointer(store, 'hybrid').unlink()
    publish(store, 'face-b-geometry', view='ribbed_face', backend='geometry')

    assert store.active(VIEW, 'hybrid') == face_a
    with pytest.raises(FileNotFoundError, match='ribbed_face'):
        store.active('ribbed_face', 'hybrid')


def test_explicit_selection_checks_context_before_replacing_the_previous_pointer(store):
    first = publish(store, 'previous-valid')
    candidate = publish(store, 'incompatible-candidate')
    store.select(VIEW, first.name, 'hybrid')
    previous = pointer(store, 'hybrid').read_bytes()
    rewrite_metadata(candidate, {'product': 'different_product'})

    with pytest.raises(ValueError, match='product'):
        store.select(VIEW, candidate.name, 'hybrid')

    assert pointer(store, 'hybrid').read_bytes() == previous
    assert store.active(VIEW, 'hybrid') == first


def test_readiness_check_does_not_hash_files_but_default_resolution_still_does(store, monkeypatch):
    directory = publish(store)
    (directory / 'payload.txt').write_text('Changed after publication', encoding='utf-8')
    real_digest = artifacts.digest
    hashed = []

    def observed_digest(path):
        hashed.append(Path(path).name)
        return real_digest(path)

    monkeypatch.setattr(artifacts, 'digest', observed_digest)

    assert store.active(VIEW, 'hybrid', verify_files=False) == directory
    assert hashed == []
    with pytest.raises(ValueError, match='integrity'):
        store.active(VIEW, 'hybrid')
    assert hashed == ['payload.txt']


@pytest.mark.parametrize('verify_files', [True, False])
def test_malformed_manifest_is_rejected_even_for_cheap_readiness(store, verify_files):
    directory = publish(store)
    atomic_json(directory / 'manifest.json', ['not', 'a', 'manifest'])

    with pytest.raises(ValueError, match='Unsupported or incomplete'):
        store.active(VIEW, 'hybrid', verify_files=verify_files)


@pytest.mark.parametrize('verify_files', [True, False])
def test_unsafe_manifest_file_paths_are_rejected_even_for_cheap_readiness(store, verify_files):
    directory = publish(store)
    rewrite_metadata(directory, {'files': {'../outside.txt': '0' * 64}})

    with pytest.raises(ValueError, match='unsafe path'):
        store.active(VIEW, 'hybrid', verify_files=verify_files)


def test_unsupported_mode_does_not_create_an_unusable_selection(store):
    directory = publish(store)

    with pytest.raises(ValueError, match='Select geometry'):
        store.select(VIEW, directory.name, 'unsupported')
    with pytest.raises(ValueError, match='Select geometry'):
        store.active(VIEW, 'unsupported')
    assert not pointer(store, 'unsupported').exists()
