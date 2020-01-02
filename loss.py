# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    cond_weight = 1.0): # Weight of the conditioning term.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in WGAN-ALP

def D_wganalp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 0.1,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fakes = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, _ = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, _ = fp32(D.get_output_for(fakes, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    alp_args_eps_min = 0.1
    alp_args_eps_max = 100

    alp_args_ip = 1
    alp_args_xi = 10

    with tf.name_scope('ALP'):
        discriminator = lambda x: fp32(D.get_output_for(x, is_training=True))[0]
        stable_norm = lambda x: tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1,2,3])), -1)
        normalize = lambda x: x / tf.maximum(tf.expand_dims(tf.expand_dims(stable_norm(x), -1), -1), 1e-10)
        d_x = lambda x, x_hat: stable_norm(x - x_hat)
        d_y = lambda y, y_hat: tf.abs(y - y_hat)

        validity_reals = real_scores_out
        d = tf.random_uniform(tf.shape(reals), 0, 1) - 0.5
        d = normalize(d)
        for _ in range(alp_args_ip):
            reals_hat = reals + alp_args_xi * d
            validity_reals_hat = discriminator(reals_hat)
            dist = tf.reduce_mean(d_y(validity_reals, validity_reals_hat))
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = normalize(tf.stop_gradient(grad))
        eps_reals = alp_args_eps_min + (alp_args_eps_max - alp_args_eps_min) * tf.random_uniform([tf.shape(reals)[0], 1, 1, 1], 0, 1)
        r_adv_reals = d * eps_reals
        reals_hat = reals + r_adv_reals

        validity_fakes = fake_scores_out
        d = tf.random_uniform(tf.shape(fakes), 0, 1) - 0.5
        d = normalize(d)
        for _ in range(alp_args_ip):
            fakes_hat = fakes + alp_args_xi * d
            validity_fakes_hat = discriminator(fakes_hat)
            dist = tf.reduce_mean(d_y(validity_fakes, validity_fakes_hat))
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = normalize(tf.stop_gradient(grad))
        eps_fakes = alp_args_eps_min + (alp_args_eps_max - alp_args_eps_min) * tf.random_uniform([tf.shape(fakes)[0], 1, 1, 1], 0, 1)
        r_adv_fakes = d * eps_fakes
        fakes_hat = fakes + r_adv_fakes

        reals_diff = d_x(reals, reals_hat)
        reals_diff = tf.maximum(reals_diff, 1e-10)
        reals_diff = tfutil.autosummary('ALP/reals_diff', reals_diff)

        validity_reals_hat = discriminator(reals_hat)
        validity_reals_diff = d_y(validity_reals, validity_reals_hat)
        validity_reals_diff = tfutil.autosummary('ALP/validity_reals_diff', validity_reals_diff)

        alp_reals = tf.maximum(validity_reals_diff / reals_diff - wgan_target, 0)
        alp_reals = tfutil.autosummary('ALP/alp_reals', alp_reals)

        nonzeros_reals = tf.greater(alp_reals, 0)
        count_reals = tf.reduce_sum(tf.cast(nonzeros_reals, tf.float32))
        count_reals = tfutil.autosummary('ALP/count_reals', count_reals)

        alp_reals_l2 = tf.square(alp_reals)
        alp_reals_l2 = tfutil.autosummary('ALP/alp_reals_l2', alp_reals_l2)
        alp_reals_l1 = tf.abs(alp_reals)
        alp_reals_l1 = tfutil.autosummary('ALP/alp_reals_l1', alp_reals_l1)
        alp_reals_loss = (alp_reals_l1 + alp_reals_l2) * wgan_lambda
        alp_reals_loss = tfutil.autosummary('ALP/alp_reals_loss', alp_reals_loss)

        fakes_diff = d_x(fakes, fakes_hat)
        fakes_diff = tf.maximum(fakes_diff, 1e-10)
        fakes_diff = tfutil.autosummary('ALP/fakes_diff', fakes_diff)

        validity_fakes_hat = discriminator(fakes_hat)
        validity_fakes_diff = d_y(validity_fakes, validity_fakes_hat)
        validity_fakes_diff = tfutil.autosummary('ALP/validity_fakes_diff', validity_fakes_diff)

        alp_fakes = tf.maximum(validity_fakes_diff / fakes_diff - wgan_target, 0)
        alp_fakes = tfutil.autosummary('ALP/alp_fakes', alp_fakes)

        nonzeros_fakes = tf.greater(alp_fakes, 0)
        count_fakes = tf.reduce_sum(tf.cast(nonzeros_fakes, tf.float32))
        count_fakes = tfutil.autosummary('ALP/count_fakes', count_fakes)

        alp_fakes_l2 = tf.square(alp_fakes)
        alp_fakes_l2 = tfutil.autosummary('ALP/alp_fakes_l2', alp_fakes_l2)
        alp_fakes_l1 = tf.abs(alp_fakes)
        alp_fakes_l1 = tfutil.autosummary('ALP/alp_fakes_l1', alp_fakes_l1)
        alp_fakes_loss = (alp_fakes_l1 + alp_fakes_l2) * wgan_lambda
        alp_fakes_loss = tfutil.autosummary('ALP/valp_fakes_loss', alp_fakes_loss)

        count = count_reals + count_fakes
        count = tfutil.autosummary('ALP/count', count)
        alp_loss = alp_reals_loss + alp_fakes_loss
        alp_loss = tfutil.autosummary('ALP/valp_loss', alp_loss)
    loss += alp_loss

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    return loss, count

#----------------------------------------------------------------------------
