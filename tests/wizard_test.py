"""Wizard tests from petrounias.org

http://www.petrounias.org/articles/2014/09/16/pickling-python-collections-with-non-built-in-type-keys-and-cycles/

Includes functionality to assist with adding compatibility to jsonpickle.

"""

import collections

from jsonpickle import decode, encode


class World:
    def __init__(self):
        self.wizards = []


class Wizard:
    def __init__(self, world, name):
        self.name = name
        self.spells = collections.OrderedDict()
        world.wizards.append(self)

    def __cmp__(self, other):
        for (ka, va), (kb, vb) in zip(self.spells.items(), other.spells.items()):
            cmp_name = cmp(ka.name, kb.name)  # noqa: F821
            if cmp_name != 0:
                print(f'Wizards cmp: {ka.name} != {kb.name}')
                return cmp_name
            for sa, sb in zip(va, vb):
                cmp_spell = cmp(sa, sb)  # noqa: F821
                if cmp_spell != 0:
                    print(f'Spells cmp: {sa.name} != {sb.name}')
                    return cmp_spell
        return cmp(self.name, other.name)  # noqa: F821

    def __eq__(self, other):
        for (ka, va), (kb, vb) in zip(self.spells.items(), other.spells.items()):
            if ka.name != kb.name:
                print(f'Wizards differ: {ka.name} != {kb.name}')
                return False
            for sa, sb in zip(va, vb):
                if sa != sb:
                    print(f'Spells differ: {sa.name} != {sb.name}')
                    return False
        return self.name == other.name

    def __hash__(self):
        return hash('Wizard %s' % self.name)


class Spell:
    def __init__(self, caster, target, name):
        self.caster = caster
        self.target = target
        self.name = name
        try:
            spells = caster.spells[target]
        except KeyError:
            spells = caster.spells[target] = []
        spells.append(self)

    def __cmp__(self, other):
        return (
            cmp(self.name, other.name)  # noqa: F821
            or cmp(self.caster.name, other.caster.name)  # noqa: F821
            or cmp(self.target.name, other.target.name)  # noqa: F821
        )  # noqa: F821

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.caster.name == other.caster.name
            and self.target.name == other.target.name
        )

    def __hash__(self):
        return hash(f'Spell {self.name} by {self.caster.name} on {self.target.name}')


def hashsum(items):
    return sum([hash(x) for x in items])


def compare_spells(a, b):
    for (ka, va), (kb, vb) in zip(a.items(), b.items()):
        if ka != kb:
            print(f'Keys differ: {ka} != {kb}')
            return False
    return True


def test_without_pickling():
    world = World()
    wizard_merlin = Wizard(world, 'Merlin')
    wizard_morgana = Wizard(world, 'Morgana')
    spell_a = Spell(wizard_merlin, wizard_morgana, 'magic-missile')
    spell_b = Spell(wizard_merlin, wizard_merlin, 'stone-skin')
    spell_c = Spell(wizard_morgana, wizard_merlin, 'geas')
    assert wizard_merlin.spells[wizard_morgana][0] == spell_a
    assert wizard_merlin.spells[wizard_merlin][0] == spell_b
    assert wizard_morgana.spells[wizard_merlin][0] == spell_c
    # Merlin has cast Magic Missile on Morgana, and Stone Skin on himself
    assert wizard_merlin.spells[wizard_morgana][0].name == 'magic-missile'
    assert wizard_merlin.spells[wizard_merlin][0].name == 'stone-skin'
    # Morgana has cast Geas on Merlin
    assert wizard_morgana.spells[wizard_merlin][0].name == 'geas'
    # Merlin's first target was Morgana
    merlin_spells = wizard_merlin.spells
    merlin_spells_keys = list(merlin_spells.keys())
    assert merlin_spells_keys[0] in wizard_merlin.spells
    assert merlin_spells_keys[0] == wizard_morgana
    # Merlin's second target was himself
    assert merlin_spells_keys[1] in wizard_merlin.spells
    assert merlin_spells_keys[1] == wizard_merlin
    # Morgana's first target was Merlin
    morgana_spells_keys = list(wizard_morgana.spells.keys())
    assert morgana_spells_keys[0] in wizard_morgana.spells
    assert morgana_spells_keys[0] == wizard_merlin
    # Merlin's first spell cast with himself as target is in the dictionary,
    # first by looking up directly with Merlin's instance object...
    assert wizard_merlin == wizard_merlin.spells[wizard_merlin][0].target
    # ...and then with the instance object directly from the dictionary keys
    assert wizard_merlin == merlin_spells[merlin_spells_keys[1]][0].target
    # Ensure Merlin's object is unique...
    assert id(wizard_merlin) == id(merlin_spells_keys[1])
    # ...and consistently hashed
    assert hash(wizard_merlin) == hash(merlin_spells_keys[1])


def test_with_pickling():
    world = World()
    wizard_merlin = Wizard(world, 'Merlin')
    wizard_morgana = Wizard(world, 'Morgana')
    wizard_morgana_prime = Wizard(world, 'Morgana')
    assert wizard_morgana.__dict__ == wizard_morgana_prime.__dict__

    spell_a = Spell(wizard_merlin, wizard_morgana, 'magic-missile')
    spell_b = Spell(wizard_merlin, wizard_merlin, 'stone-skin')
    spell_c = Spell(wizard_morgana, wizard_merlin, 'geas')
    assert wizard_merlin.spells[wizard_morgana][0] == spell_a
    assert wizard_merlin.spells[wizard_merlin][0] == spell_b
    assert wizard_morgana.spells[wizard_merlin][0] == spell_c

    flat_world = encode(world, keys=True)
    u_world = decode(flat_world, keys=True)
    u_wizard_merlin = u_world.wizards[0]
    u_wizard_morgana = u_world.wizards[1]
    morgana_spells_encoded = encode(wizard_morgana.spells, keys=True)
    morgana_spells_decoded = decode(morgana_spells_encoded, keys=True)
    assert wizard_morgana.spells == morgana_spells_decoded

    morgana_encoded = encode(wizard_morgana, keys=True)
    morgana_decoded = decode(morgana_encoded, keys=True)
    assert wizard_morgana == morgana_decoded
    assert hash(wizard_morgana) == hash(morgana_decoded)
    assert wizard_morgana.spells == morgana_decoded.spells
    # Merlin has cast Magic Missile on Morgana, and Stone Skin on himself
    merlin_spells = u_wizard_merlin.spells
    assert merlin_spells[u_wizard_morgana][0].name == 'magic-missile'
    assert merlin_spells[u_wizard_merlin][0].name == 'stone-skin'
    # Morgana has cast Geas on Merlin
    assert u_wizard_morgana.spells[u_wizard_merlin][0].name == 'geas'
    # Merlin's first target was Morgana
    merlin_spells_keys = list(u_wizard_merlin.spells.keys())
    assert merlin_spells_keys[0] in u_wizard_merlin.spells
    assert merlin_spells_keys[0] == u_wizard_morgana
    # Merlin's second target was himself
    assert merlin_spells_keys[1] in u_wizard_merlin.spells
    assert merlin_spells_keys[1] == u_wizard_merlin
    # Morgana's first target was Merlin
    morgana_spells_keys = list(u_wizard_morgana.spells.keys())
    assert morgana_spells_keys[0] in u_wizard_morgana.spells
    assert morgana_spells_keys[0] == u_wizard_merlin
    # Merlin's first spell cast with himself as target is in the dict.
    # First try the lookup with Merlin's instance object
    assert u_wizard_merlin == merlin_spells[u_wizard_merlin][0].target
    # Next try the lookup with the object from the dictionary keys.
    assert u_wizard_merlin == merlin_spells[merlin_spells_keys[1]][0].target
    # Ensure Merlin's object is unique and consistently hashed.
    assert id(u_wizard_merlin) == id(merlin_spells_keys[1])
    assert hash(u_wizard_merlin) == hash(merlin_spells_keys[1])
